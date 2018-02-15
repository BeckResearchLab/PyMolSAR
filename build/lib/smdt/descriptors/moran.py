
from rdkit import Chem
from smdt.descriptors import AtomProperty
import pandas as pd
import numpy


def _CalculateMoranAutocorrelation(mol, lag=1, propertylabel='m'):
    """
    **Internal used only**
    Calculation of Moran autocorrelation descriptors based on
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
                res = res + (temp1 - aveweight) * (temp2 - aveweight)
                index = index + 1
            else:
                res = res + 0.0

    if sum(tempp) == 0 or index == 0:
        result = 0
    else:
        result = (res / index) / (sum(tempp) / Natom)
    return round(result, 3)


def CalculateMoranAutoMass(mol):
    """
    Calculation of Moran autocorrelation descriptors based on
    carbon-scaled atomic mass.
    """
    res = {}
    for i in range(8):
        res['MATSm' + str(i + 1)] = _CalculateMoranAutocorrelation(mol, lag=i + 1, propertylabel='m')
    return res


def CalculateMoranAutoVolume(mol):
    """
    Calculation of Moran autocorrelation descriptors based on
    carbon-scaled atomic van der Waals volume.
    """
    res = {}

    for i in range(8):
        res['MATSv' + str(i + 1)] = _CalculateMoranAutocorrelation(mol, lag=i + 1, propertylabel='V')

    return res


def CalculateMoranAutoElectronegativity(mol):
    """
    Calculation of Moran autocorrelation descriptors based on
    carbon-scaled atomic Sanderson electronegativity.
    """
    res = {}
    for i in range(8):
        res['MATSe' + str(i + 1)] = _CalculateMoranAutocorrelation(mol, lag=i + 1, propertylabel='En')
    return res


def CalculateMoranAutoPolarizability(mol):
    """
    Calculation of Moran autocorrelation descriptors based on
    carbon-scaled atomic polarizability.
    """
    res = {}

    for i in range(8):
        res['MATSp' + str(i + 1)] = _CalculateMoranAutocorrelation(mol, lag=i + 1, propertylabel='alapha')

    return res


def GetMoranAutoofMol(mol):
    """
    Calcualate all Moran autocorrelation descriptors.
    (carbon-scaled atomic mass, carbon-scaled atomic van der Waals volume,
    carbon-scaled atomic Sanderson electronegativity, carbon-scaled atomic polarizability)
    """
    res = {}
    res.update(CalculateMoranAutoMass(mol))
    res.update(CalculateMoranAutoVolume(mol))
    res.update(CalculateMoranAutoElectronegativity(mol))
    res.update(CalculateMoranAutoPolarizability(mol))

    return res

def getMoranAuto(df_x):
    """
    Calculates all Moran Auto-correlation descriptors for the dataset
        Parameters:
            df_x: pandas.DataFrame
                SMILES DataFrame
        Returns:
            moran_descriptors: pandas.DataFrame
                Moran Auto-correlation Descriptors DataFrame
    """

    labels = []
    for i in range(8):
        labels.append('MATSm' + str(i + 1))
        labels.append('MATSv' + str(i + 1))
        labels.append('MATSe' + str(i + 1))
        labels.append('MATSp' + str(i + 1))
    r = {}
    for key in labels:
        r[key] = []
    i=0
    for m in df_x['SMILES']:
        i=i+1
        mol = Chem.MolFromSmiles(m)
        res = GetMoranAutoofMol(mol)
        for key in labels:
            r[key].append(res[key])
    moran_descriptors = pd.DataFrame(r).round(3)
    return pd.DataFrame(moran_descriptors)