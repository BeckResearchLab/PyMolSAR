
from rdkit import Chem
from rdkit.Chem import rdchem
#from rdkit.Chem import pyPeriodicTable as PeriodicTable
periodicTable = Chem.GetPeriodicTable()
import pandas as pd


hallKierAlphas = {'Br': [None, None, 0.48],
                  'C': [-0.22, -0.13, 0.0],
                  'Cl': [None, None, 0.29],
                  'F': [None, None, -0.07],
                  'H': [0.0, 0.0, 0.0],
                  'I': [None, None, 0.73],
                  'N': [-0.29, -0.2, -0.04],
                  'O': [None, -0.2, -0.04],
                  'P': [None, 0.3, 0.43],
                  'S': [None, 0.22, 0.35]}


def CalculateKappa1(mol):
    """
    Calculation of molecular shape index for one bonded fragment
    """
    P1 = mol.GetNumBonds(onlyHeavy=1)
    A = mol.GetNumHeavyAtoms()
    denom = P1 + 0.0
    if denom:
        kappa = (A) * (A - 1) ** 2 / denom ** 2
    else:
        kappa = 0.0
    return round(kappa, 3)


def CalculateKappa2(mol):
    """
    Calculation of molecular shape index for two bonded fragment
    """
    P2 = len(Chem.FindAllPathsOfLengthN(mol, 2))
    A = mol.GetNumHeavyAtoms()

    denom = P2 + 0.0
    if denom:
        kappa = (A - 1) * (A - 2) ** 2 / denom ** 2
    else:
        kappa = 0.0
    return round(kappa, 3)


def CalculateKappa3(mol):
    """
    Calculation of molecular shape index for three bonded fragment
    """
    P3 = len(Chem.FindAllPathsOfLengthN(mol, 3))
    A = mol.GetNumHeavyAtoms()

    denom = P3 + 0.0
    if denom:
        if A % 2 == 1:
            kappa = (A - 1) * (A - 3) ** 2 / denom ** 2
        else:
            kappa = (A - 3) * (A - 2) ** 2 / denom ** 2
    else:
        kappa = 0.0
    return round(kappa, 3)


def _HallKierAlpha(mol):
    """
    *Internal Use Only*
    Calculation of the Hall-Kier alpha value for a molecule
    """
    alphaSum = 0.0
    rC = periodicTable.GetRb0(6)
    for atom in mol.GetAtoms():
        atNum = atom.GetAtomicNum()
        if not atNum: continue
        symb = atom.GetSymbol()
        alphaV = hallKierAlphas.get(symb, None)
        if alphaV is not None:
            hyb = atom.GetHybridization() - 2
            if hyb < len(alphaV):
                alpha = alphaV[hyb]
                if alpha is None:
                    alpha = alphaV[-1]
            else:
                alpha = alphaV[-1]
        else:
            rA = periodicTable.GetRb0(atNum)
            alpha = rA / rC - 1
        alphaSum += alpha
    return alphaSum


def CalculateKappaAlapha1(mol):
    """
    Calculation of molecular shape index for one bonded fragment
    """
    P1 = mol.GetNumBonds(onlyHeavy=1)
    A = mol.GetNumHeavyAtoms()
    alpha = _HallKierAlpha(mol)
    denom = P1 + alpha
    if denom:
        kappa = (A + alpha) * (A + alpha - 1) ** 2 / denom ** 2
    else:
        kappa = 0.0
    return round(kappa, 3)


def CalculateKappaAlapha2(mol):
    """
    Calculation of molecular shape index for two bonded fragment
    """
    P2 = len(Chem.FindAllPathsOfLengthN(mol, 2))
    A = mol.GetNumHeavyAtoms()
    alpha = _HallKierAlpha(mol)
    denom = P2 + alpha
    if denom:
        kappa = (A + alpha - 1) * (A + alpha - 2) ** 2 / denom ** 2
    else:
        kappa = 0.0
    return round(kappa, 3)


def CalculateKappaAlapha3(mol):
    """
    Calculation of molecular shape index for three bonded fragment
    """
    P3 = len(Chem.FindAllPathsOfLengthN(mol, 3))
    A = mol.GetNumHeavyAtoms()
    alpha = _HallKierAlpha(mol)
    denom = P3 + alpha
    if denom:
        if A % 2 == 1:
            kappa = (A + alpha - 1) * (A + alpha - 3) ** 2 / denom ** 2
        else:
            kappa = (A + alpha - 3) * (A + alpha - 2) ** 2 / denom ** 2
    else:
        kappa = 0.0
    return round(kappa, 3)


def CalculateFlexibility(mol):
    """
    Calculation of Kier molecular flexibility index
    """
    kappa1 = CalculateKappaAlapha1(mol)
    kappa2 = CalculateKappaAlapha2(mol)
    A = mol.GetNumHeavyAtoms()
    phi = kappa1 * kappa2 / (A + 0.0)
    return phi


def GetKappaofMol(mol):
    """
    Calculation of all kappa values.
    """
    res = {}
    res['kappa1'] = CalculateKappa1(mol)
    res['kappa2'] = CalculateKappa2(mol)
    res['kappa3'] = CalculateKappa3(mol)
    res['kappam1'] = CalculateKappaAlapha1(mol)
    res['kappam2'] = CalculateKappaAlapha2(mol)
    res['kappam3'] = CalculateKappaAlapha3(mol)
    res['phi'] = CalculateFlexibility(mol)
    return res

def getKappa(df_x):
    """
    Calculates all Kappa descriptors for the dataset
        Parameters:
            df_x: pandas.DataFrame
                SMILES DataFrame
        Returns:
            kappa_descriptors: pandas.DataFrame
                Kappa Descriptors DataFrame
    """
    r = {}
    labels = ['kappa1','kappa2','kappa3','kappam1','kappam2','kappam3','phi']
    for key in labels:
        r[key] = []
    for m in df_x['SMILES']:
        mol = Chem.MolFromSmiles(m)
        res = GetKappaofMol(mol)
        for key in labels:
            r[key].append(res[key])
    kappa_descriptors = pd.DataFrame(r).round(3)
    return pd.DataFrame(kappa_descriptors)