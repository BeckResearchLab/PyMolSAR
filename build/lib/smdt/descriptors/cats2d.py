
import scipy
from rdkit import Chem
import pandas as pd

PPP = {"D": ['[OH]', '[#7H,#7H2]'],
       "A": ['[O]', '[#7H0]'],
       'P': ['[*+]', '[#7H2]'],
       'N': ['[*-]', '[C&$(C(=O)O)]', '[P&$(P(=O)O)]', '[S&$(S(=O)O)]'],
       "L": ['[Cl,Br,I]', '[S;D2;$(S(C)(C))]']}

def MatchAtomType(IndexList, AtomTypeDict):
    """
    Mapping two atoms with a certain distance into their atom types
    such as AA,AL, DP,LD etc. The result is a list format.
    """
    First = []
    Second = []
    for i in AtomTypeDict:
        if IndexList[0] in AtomTypeDict[i]:
            First.append(i)
        if IndexList[1] in AtomTypeDict[i]:
            Second.append(i)

    temp = []
    for i in First:
        for j in Second:
            temp.append(i + j)

    temp1 = []
    for i in temp:
        if i in ['AD', 'PD', 'ND', 'LD', 'PA', 'NA', 'LA', 'NP', 'LN', 'LP']:
            temp1.append(i[1] + i[0])
        else:
            temp1.append(i)

    res = []
    for i in temp1:
        if i not in res:
            res.append(i)

    return res


def ContructLFromGraphSearch(mol):
    """
    The last lipophilic pattern on page 55 of the book is realized as a graph
    search and not as a SMARTS search. "L" carbon atom adjacent only to carbon atoms.
    The result is a list format.
    """

    AtomIndex = []
    Hmol = Chem.RemoveHs(mol)
    for atom in Hmol.GetAtoms():
        temp = []
        if atom.GetAtomicNum() == 6:
            for neighatom in atom.GetNeighbors():
                if neighatom.GetAtomicNum() == 6:
                    temp.append(0)
                elif neighatom.GetAtomicNum() == 1:
                    continue
                else:
                    temp.append(1)
            if sum(temp) == 0:
                AtomIndex.append(atom.GetIdx())

    return AtomIndex


def FormCATSLabel(PathLength=10):
    """
    Construct the CATS label such as AA0, AA1,....AP3,.......
    The result is a list format.
    A   acceptor;
    P   positive;
    N   negative;
    L   lipophilic;
    D   donor;
    """
    AtomPair = ['DD', 'DA', 'DP', 'DN', 'DL', 'AA', 'AP', 'AN', 'AL',
                'PP', 'PN', 'PL', 'NN', 'NL', 'LL']
    CATSLabel = []
    for i in AtomPair:
        for k in range(PathLength):
            CATSLabel.append("CATS_" + i + str(k))
    return CATSLabel


def FormCATSDict(AtomDict, CATSLabel):
    """
    Construct the CATS dict. The result is a dict format.
    """

    temp = []
    for i in AtomDict:
        for j in AtomDict[i]:
            if len(j) == 0:
                continue
            else:
                temp.append(j + str(i))

    res = dict()
    for i in set(temp):
        res.update({"CATS_" + i: temp.count(i)})

    result = dict(zip(CATSLabel, [0 for i in CATSLabel]))
    result.update(res)

    return result


def AssignAtomType(mol):
    """
    Assign the atoms in the mol object into each of the PPP type
    according to PPP list definition.
    Note: res is a dict form such as {'A': [2], 'P': [], 'N': [4]}
    """
    res = dict()
    for ppptype in PPP:
        temp = []
        for i in PPP[ppptype]:
            patt = Chem.MolFromSmarts(i)
            atomindex = mol.GetSubstructMatches(patt)
            atomindex = [k[0] for k in atomindex]
            temp.extend(atomindex)
        res.update({ppptype: temp})
    temp = ContructLFromGraphSearch(mol)
    temp.extend(res['L'])
    res.update({'L': temp})

    return res


def CATS2DforMol(mol, PathLength=10, scale=3):
    """
    The main program for calculating the CATS descriptors.
    CATS: chemically advanced template search
    PathLength is the max topological distance between two atoms.
    Scale is the normalization method (descriptor scaling method)
    Scale = 1 indicates that no normalization. That is to say: the
    values of the vector represent raw counts ("counts").
    Scale = 2 indicates that division by the number of non-hydrogen
    atoms (heavy atoms) in the molecule.
    Scale = 3 indicates that division of each of 15 possible PPP pairs
    by the added occurrences of the two respective PPPs.
    """
    Hmol = Chem.RemoveHs(mol)
    AtomNum = Hmol.GetNumAtoms()
    atomtype = AssignAtomType(Hmol)
    DistanceMatrix = Chem.GetDistanceMatrix(Hmol)
    DM = scipy.triu(DistanceMatrix)
    tempdict = {}
    for PL in range(0, PathLength):
        if PL == 0:
            Index = [[k, k] for k in range(AtomNum)]
        else:
            Index1 = scipy.argwhere(DM == PL)
            Index = [[k[0], k[1]] for k in Index1]
        temp = []
        for j in Index:
            temp.extend(MatchAtomType(j, atomtype))
        tempdict.update({PL: temp})

    CATSLabel = FormCATSLabel(PathLength)
    CATS1 = FormCATSDict(tempdict, CATSLabel)

    AtomPair = ['DD', 'DA', 'DP', 'DN', 'DL', 'AA', 'AP',
                'AN', 'AL', 'PP', 'PN', 'PL', 'NN', 'NL', 'LL']
    temp = []
    for i, j in tempdict.items():
        temp.extend(j)

    AtomPairNum = {}
    for i in AtomPair:
        AtomPairNum.update({i: temp.count(i)})
    CATS = {}
    if scale == 1:
        CATS = CATS1
    if scale == 2:
        for i in CATS1:
            CATS.update({i: round(CATS1[i] / (AtomNum + 0.0), 3)})
    if scale == 3:
        for i in CATS1:
            if AtomPairNum[i[5:7]] == 0:
                CATS.update({i: round(CATS1[i], 3)})
            else:
                CATS.update({i: round(CATS1[i] / (AtomPairNum[i[5:7]] + 0.0), 3)})

    return CATS

_labels = FormCATSLabel()

def getCATS2D(df_x):
    """
    Calculates all CATS 2D descriptors for the dataset
        Parameters:
            df_x: pandas.DataFrame
                SMILES DataFrame
        Returns:
            cats2d_descriptors: pandas.DataFrame
                CATS 2D Descriptors DataFrame
    """
    r = {}
    for key in _labels:
       r[key] = []
    for m in df_x['SMILES']:
        mol = Chem.MolFromSmiles(m)
        res = CATS2DforMol(mol)
        for key in _labels:
            r[key].append(res[key])
    cats2d_descriptors = pd.DataFrame(r).round(3)
    return pd.DataFrame(cats2d_descriptors)