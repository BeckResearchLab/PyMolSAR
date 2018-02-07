from rdkit import Chem
import numpy
import copy
import pandas as pd

def _CalculateEntropy(Probability):
    """
    **Internal used only**
    Calculation of entropy (Information content) for probability given
    """
    res = 0.0
    for i in Probability:
        if i != 0:
            res = res - i * numpy.log2(i)

    return res


def CalculateBasakIC0(mol):
    """
    Obtain the information content with order 0 proposed by Basak
    """

    BasakIC = 0.0
    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = []
    for i in range(nAtoms):
        at = Hmol.GetAtomWithIdx(i)
        IC.append(at.GetAtomicNum())
    Unique = numpy.unique(IC)
    NAtomType = len(Unique)
    NTAtomType = numpy.zeros(NAtomType, numpy.float)
    for i in range(NAtomType):
        NTAtomType[i] = IC.count(Unique[i])

    if nAtoms != 0:
        BasakIC = _CalculateEntropy(NTAtomType / nAtoms)
    else:
        BasakIC = 0.0

    return BasakIC


def CalculateBasakSIC0(mol):
    """
    Obtain the structural information content with order 0
    proposed by Basak
    """

    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = CalculateBasakIC0(mol)
    if nAtoms <= 1:
        BasakSIC = 0.0
    else:
        BasakSIC = IC / numpy.log2(nAtoms)

    return BasakSIC


def CalculateBasakCIC0(mol):
    """
    Obtain the complementary information content with order 0
    proposed by Basak
    """
    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = CalculateBasakIC0(mol)
    if nAtoms <= 1:
        BasakCIC = 0.0
    else:
        BasakCIC = numpy.log2(nAtoms) - IC

    return BasakCIC


def _CalculateBasakICn(mol, NumPath=1):
    """
    **internal used only**
    Obtain the information content with order n proposed by Basak
    """
    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    TotalPath = Chem.FindAllPathsOfLengthN(Hmol, NumPath, useBonds=0, useHs=1)
    if len(TotalPath) == 0:
        BasakIC = 0.0
    else:
        IC = {}
        for i in range(nAtoms):
            temp = []
            at = Hmol.GetAtomWithIdx(i)
            temp.append(at.GetAtomicNum())
            for index in TotalPath:
                if i == index[0]:
                    temp.append([Hmol.GetAtomWithIdx(kk).GetAtomicNum() for kk in index[1:]])
                if i == index[-1]:
                    cds = list(index)
                    cds.reverse()
                    temp.append([Hmol.GetAtomWithIdx(kk).GetAtomicNum() for kk in cds[1:]])

            IC[str(i)] = temp
        cds = []
        for value in IC.values():
            for i in range(len(value)):
                if isinstance(value[i],list):
                    value[i] = value[i][0]
            cds.append(sorted(list(value)))
        kkk = list(range(len(cds)))
        aaa = copy.deepcopy(kkk)
        res = []
        for i in aaa:
            if i in kkk:
                jishu = 0
                kong = []
                temp1 = cds[i]
                for j in aaa:
                    if cds[j] == temp1:
                        jishu = jishu + 1
                        kong.append(j)
                for ks in kong:
                    kkk.remove(ks)
                res.append(jishu)

        BasakIC = _CalculateEntropy(numpy.array(res, numpy.float) / sum(res))

    return BasakIC


def CalculateBasakIC1(mol):
    """
    Obtain the information content with order 1 proposed by Basak
    """
    return _CalculateBasakICn(mol, NumPath=2)


def CalculateBasakIC2(mol):
    """
    Obtain the information content with order 2 proposed by Basak
    """
    return _CalculateBasakICn(mol, NumPath=3)


def CalculateBasakIC3(mol):
    """
    Obtain the information content with order 3 proposed by Basak
    """
    return _CalculateBasakICn(mol, NumPath=4)


def CalculateBasakIC4(mol):
    """
    Obtain the information content with order 4 proposed by Basak
    """
    return _CalculateBasakICn(mol, NumPath=5)


def CalculateBasakIC5(mol):
    """
    Obtain the information content with order 5 proposed by Basak
    """
    return _CalculateBasakICn(mol, NumPath=6)


def CalculateBasakIC6(mol):
    """
    Obtain the information content with order 6 proposed by Basak
    """
    return _CalculateBasakICn(mol, NumPath=7)


def CalculateBasakSIC1(mol):
    """
    Obtain the structural information content with order 1
    proposed by Basak.
    """
    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = CalculateBasakIC1(mol)
    if nAtoms <= 1:
        BasakSIC = 0.0
    else:
        BasakSIC = IC / numpy.log2(nAtoms)

    return BasakSIC


def CalculateBasakSIC2(mol):
    """
    Obtain the structural information content with order 2 proposed
    by Basak.
    """
    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = CalculateBasakIC2(mol)
    if nAtoms <= 1:
        BasakSIC = 0.0
    else:
        BasakSIC = IC / numpy.log2(nAtoms)

    return BasakSIC


def CalculateBasakSIC3(mol):
    """
    Obtain the structural information content with order 3 proposed
    by Basak.
    """
    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = CalculateBasakIC3(mol)
    if nAtoms <= 1:
        BasakSIC = 0.0
    else:
        BasakSIC = IC / numpy.log2(nAtoms)

    return BasakSIC


def CalculateBasakSIC4(mol):
    """
    Obtain the structural information content with order 4 proposed
    by Basak.
    """
    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = CalculateBasakIC4(mol)
    if nAtoms <= 1:
        BasakSIC = 0.0
    else:
        BasakSIC = IC / numpy.log2(nAtoms)

    return BasakSIC


def CalculateBasakSIC5(mol):
    """
    Obtain the structural information content with order 5 proposed
    by Basak.
    """
    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = CalculateBasakIC5(mol)
    if nAtoms <= 1:
        BasakSIC = 0.0
    else:
        BasakSIC = IC / numpy.log2(nAtoms)

    return BasakSIC


def CalculateBasakSIC6(mol):
    """
    Obtain the structural information content with order 6 proposed
    by Basak.
    """
    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = CalculateBasakIC6(mol)
    if nAtoms <= 1:
        BasakSIC = 0.0
    else:
        BasakSIC = IC / numpy.log2(nAtoms)

    return BasakSIC


def CalculateBasakCIC1(mol):
    """
    Obtain the complementary information content with order 1 proposed
    by Basak.
    """
    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = CalculateBasakIC1(mol)
    if nAtoms <= 1:
        BasakCIC = 0.0
    else:
        BasakCIC = numpy.log2(nAtoms) - IC
    return BasakCIC


def CalculateBasakCIC2(mol):
    """
    Obtain the complementary information content with order 2 proposed
    by Basak.
    """
    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = CalculateBasakIC2(mol)
    if nAtoms <= 1:
        BasakCIC = 0.0
    else:
        BasakCIC = numpy.log2(nAtoms) - IC

    return BasakCIC


def CalculateBasakCIC3(mol):
    """
    Obtain the complementary information content with order 3 proposed
    by Basak.
    """
    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = CalculateBasakIC3(mol)
    if nAtoms <= 1:
        BasakCIC = 0.0
    else:
        BasakCIC = numpy.log2(nAtoms) - IC

    return BasakCIC


def CalculateBasakCIC4(mol):
    """
    Obtain the complementary information content with order 4 proposed
    by Basak.
    """
    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = CalculateBasakIC4(mol)
    if nAtoms <= 1:
        BasakCIC = 0.0
    else:
        BasakCIC = numpy.log2(nAtoms) - IC

    return BasakCIC


def CalculateBasakCIC5(mol):
    """
    Obtain the complementary information content with order 5 proposed
    by Basak.
    """
    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = CalculateBasakIC5(mol)
    if nAtoms <= 1:
        BasakCIC = 0.0
    else:
        BasakCIC = numpy.log2(nAtoms) - IC

    return BasakCIC


def CalculateBasakCIC6(mol):
    """
    Obtain the complementary information content with order 6 proposed
    by Basak.
    """
    Hmol = Chem.AddHs(mol)
    nAtoms = Hmol.GetNumAtoms()
    IC = CalculateBasakIC6(mol)
    if nAtoms <= 1:
        BasakCIC = 0.0
    else:
        BasakCIC = numpy.log2(nAtoms) - IC

    return BasakCIC


_basak = {'CIC0': CalculateBasakCIC0,
          'CIC1': CalculateBasakCIC1,
          'CIC2': CalculateBasakCIC2,
          'CIC3': CalculateBasakCIC3,
          'CIC4': CalculateBasakCIC4,
          'CIC5': CalculateBasakCIC5,
          'CIC6': CalculateBasakCIC6,
          'SIC0': CalculateBasakSIC0,
          'SIC1': CalculateBasakSIC1,
          'SIC2': CalculateBasakSIC2,
          'SIC3': CalculateBasakSIC3,
          'SIC4': CalculateBasakSIC4,
          'SIC5': CalculateBasakSIC5,
          'SIC6': CalculateBasakSIC6,
          'IC0': CalculateBasakIC0,
          'IC1': CalculateBasakIC1,
          'IC2': CalculateBasakIC2,
          'IC3': CalculateBasakIC3,
          'IC4': CalculateBasakIC4,
          'IC5': CalculateBasakIC5,
          'IC6': CalculateBasakIC6
          }


def GetBasakofMol(mol):
    """
    Get the dictionary of basak descriptors for given moelcule mol
    """
    result = {}
    for DesLabel in _basak.keys():
        result[DesLabel] = round(_basak[DesLabel](mol), 3)
    return result


def getBasak(df_x):
    """
    Calculates all Basak descriptors for the dataset
        Parameters:
            df_x: pandas.DataFrame
                SMILES DataFrame
        Returns:
            basak_descriptors: pandas.DataFrame
                Basak Descriptors DataFrame
    """
    r = {}
    for key in _basak:
        r[key] = []
    for m in df_x['SMILES']:
        mol = Chem.MolFromSmiles(m)
        res = GetBasakofMol(mol)
        for key in _basak:
            r[key].append(res[key])
    basak_descriptors = pd.DataFrame(r).round(3)
    return pd.DataFrame(basak_descriptors)