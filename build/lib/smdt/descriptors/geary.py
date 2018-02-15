
from rdkit import Chem
from smdt.descriptors import AtomProperty
import numpy
import pandas as pd

def _CalculateGearyAutocorrelation(mol, lag=1, propertylabel='m'):
    """
    **Internal used only**
    Calculation of Geary autocorrelation descriptors based on
    different property weights.
    """

    Natom = mol.GetNumAtoms()

    prolist = []
    for i in mol.GetAtoms():
        temp = AtomProperty.GetRelativeAtomicProperty(i.GetSymbol(), propertyname=propertylabel)
        prolist.append(temp)

    aveweight = sum(prolist) / Natom

    tempp = [numpy.square(x - aveweight) for x in prolist]

    GetDistanceMatrix = Chem.GetDistanceMatrix(mol)
    res = 0.0
    index = 0
    for i in range(Natom):
        for j in range(Natom):
            if GetDistanceMatrix[i, j] == lag:
                atom1 = mol.GetAtomWithIdx(i)
                atom2 = mol.GetAtomWithIdx(j)
                temp1 = AtomProperty.GetRelativeAtomicProperty(element=atom1.GetSymbol(), propertyname=propertylabel)
                temp2 = AtomProperty.GetRelativeAtomicProperty(element=atom2.GetSymbol(), propertyname=propertylabel)
                res = res + numpy.square(temp1 - temp2)
                index = index + 1
            else:
                res = res + 0.0

    if sum(tempp) == 0 or index == 0:
        result = 0
    else:
        result = (res / index / 2) / (sum(tempp) / (Natom - 1))

    return round(result, 3)


def CalculateGearyAutoMass(mol):
    """
    Calculation of Geary autocorrelation descriptors based on
    carbon-scaled atomic mass.
    """
    res = {}

    for i in range(8):
        res['GATSm' + str(i + 1)] = _CalculateGearyAutocorrelation(mol, lag=i + 1, propertylabel='m')

    return res


def CalculateGearyAutoVolume(mol):
    """
    Calculation of Geary autocorrelation descriptors based on
    carbon-scaled atomic van der Waals volume.
    """
    res = {}

    for i in range(8):
        res['GATSv' + str(i + 1)] = _CalculateGearyAutocorrelation(mol, lag=i + 1, propertylabel='V')

    return res


def CalculateGearyAutoElectronegativity(mol):
    """
    Calculation of Geary autocorrelation descriptors based on
    carbon-scaled atomic Sanderson electronegativity.
    """
    res = {}

    for i in range(8):
        res['GATSe' + str(i + 1)] = _CalculateGearyAutocorrelation(mol, lag=i + 1, propertylabel='En')

    return res


def CalculateGearyAutoPolarizability(mol):
    """
    Calculation of Geary autocorrelation descriptors based on
    carbon-scaled atomic polarizability.
    """
    res = {}

    for i in range(8):
        res['GATSp' + str(i + 1)] = _CalculateGearyAutocorrelation(mol, lag=i + 1, propertylabel='alapha')

    return res


def GetGearyAutoofMol(mol):
    """
    Calcualate all Geary autocorrelation descriptors.
    """
    res = {}
    res.update(CalculateGearyAutoMass(mol))
    res.update(CalculateGearyAutoVolume(mol))
    res.update(CalculateGearyAutoElectronegativity(mol))
    res.update(CalculateGearyAutoPolarizability(mol))

    return res


def getGearyAuto(df_x):
    """
    Calculates all Geary Auto-correlation descriptors for the dataset
        Parameters:
            df_x: pandas.DataFrame
                SMILES DataFrame
        Returns:
            geary_descriptors: pandas.DataFrame
                Geary Auto-correlation Descriptors DataFrame
    """

    labels = []
    for i in range(8):
        labels.append('GATSm' + str(i + 1))
        labels.append('GATSv' + str(i + 1))
        labels.append('GATSe' + str(i + 1))
        labels.append('GATSp' + str(i + 1))
    r = {}
    for key in labels:
        r[key] = []
    for m in df_x['SMILES']:
        mol = Chem.MolFromSmiles(m)
        res = GetGearyAutoofMol(mol)
        for key in labels:
            r[key].append(res[key])
    geary_descriptors = pd.DataFrame(r).round(3)
    return pd.DataFrame(geary_descriptors)