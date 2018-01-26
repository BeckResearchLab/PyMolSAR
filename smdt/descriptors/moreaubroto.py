"""
The calculation of Moreau-Broto autocorrelation descriptors. You can get 32
molecular decriptors.
"""

from rdkit import Chem
from smdt.descriptors import AtomProperty
import pandas as pd
import numpy

def _CalculateMoreauBrotoAutocorrelation(mol, lag=1, propertylabel='m'):
    """
    #################################################################
    **Internal used only**

    Calculation of Moreau-Broto autocorrelation descriptors based on

    different property weights.

    Usage:

    res=_CalculateMoreauBrotoAutocorrelation(mol, lag=1,propertylabel='m')

    Input: mol is a molecule object.

    lag is the topological distance between atom i and atom j.

    propertylabel is the weighted property.

    Output: res is a numeric value.
    #################################################################
    """

    Natom = mol.GetNumAtoms()

    GetDistanceMatrix = Chem.GetDistanceMatrix(mol)
    res = 0.0
    for i in range(Natom):
        for j in range(Natom):
            if GetDistanceMatrix[i, j] == lag:
                atom1 = mol.GetAtomWithIdx(i)
                atom2 = mol.GetAtomWithIdx(j)
                temp1 = AtomProperty.GetRelativeAtomicProperty(element=atom1.GetSymbol(), propertyname=propertylabel)
                temp2 = AtomProperty.GetRelativeAtomicProperty(element=atom2.GetSymbol(), propertyname=propertylabel)
                res = res + temp1 * temp2
            else:
                res = res + 0.0

    return round(numpy.log(res / 2 + 1), 3)


def CalculateMoreauBrotoAutoMass(mol):
    """
    #################################################################
    Calculation of Moreau-Broto autocorrelation descriptors based on

    carbon-scaled atomic mass.

    Usage:

    res=CalculateMoreauBrotoAutoMass(mol)

    Input: mol is a molecule object.

    Output: res is a dict form containing eight moreau broto autocorrealtion

    descriptors.
    #################################################################
    """
    res = {}

    for i in range(8):
        res['ATSm' + str(i + 1)] = _CalculateMoreauBrotoAutocorrelation(mol, lag=i + 1, propertylabel='m')

    return res


def CalculateMoreauBrotoAutoVolume(mol):
    """
    #################################################################
    Calculation of Moreau-Broto autocorrelation descriptors based on

    carbon-scaled atomic van der Waals volume.

    Usage:

    res=CalculateMoreauBrotoAutoVolume(mol)

    Input: mol is a molcule object.

    Output: res is a dict form containing eight moreau broto autocorrealtion

    descriptors.
    #################################################################
    """
    res = {}

    for i in range(8):
        res['ATSv' + str(i + 1)] = _CalculateMoreauBrotoAutocorrelation(mol, lag=i + 1, propertylabel='V')

    return res


def CalculateMoreauBrotoAutoElectronegativity(mol):
    """
    #################################################################
    Calculation of Moreau-Broto autocorrelation descriptors based on

    carbon-scaled atomic Sanderson electronegativity.
    Usage:

    res=CalculateMoreauBrotoAutoElectronegativity(mol)

    Input: mol is a molcule object.

    Output: res is a dict form containing eight moreau broto autocorrealtion

    descriptors.
    #################################################################
    """
    res = {}

    for i in range(8):
        res['ATSe' + str(i + 1)] = _CalculateMoreauBrotoAutocorrelation(mol, lag=i + 1, propertylabel='En')

    return res


def CalculateMoreauBrotoAutoPolarizability(mol):
    """
    #################################################################
    Calculation of Moreau-Broto autocorrelation descriptors based on

    carbon-scaled atomic polarizability.
    res=CalculateMoreauBrotoAutoPolarizability(mol)

    Input: mol is a molcule object.

    Output: res is a dict form containing eight moreau broto autocorrealtion

    descriptors.
    #################################################################
    """
    res = {}

    for i in range(8):
        res['ATSp' + str(i + 1)] = _CalculateMoreauBrotoAutocorrelation(mol, lag=i + 1, propertylabel='alapha')

    return res


def GetMoreauBrotoAutoofMol(mol):
    """
    #################################################################
    Calcualate all Moreau-Broto autocorrelation descriptors.

    (carbon-scaled atomic mass, carbon-scaled atomic van der Waals volume,

    carbon-scaled atomic Sanderson electronegativity,

    carbon-scaled atomic polarizability)

    Usage:

    res=GetMoreauBrotoAuto(mol)

    Input: mol is a molecule object.

    Output: res is a dict form containing all moreau broto autocorrelation

    descriptors.
    #################################################################
    """
    res = {}
    res.update(CalculateMoreauBrotoAutoMass(mol))
    res.update(CalculateMoreauBrotoAutoVolume(mol))
    res.update(CalculateMoreauBrotoAutoElectronegativity(mol))
    res.update(CalculateMoreauBrotoAutoPolarizability(mol))

    return res

def GetMoreauBrotoAuto(df_x):
    r = {}
    labels = []
    for i in range(8):
        labels.append('ATSm' + str(i + 1))
        labels.append('ATSv' + str(i + 1))
        labels.append('ATSe' + str(i + 1))
        labels.append('ATSp' + str(i + 1))
    for key in labels:
        r[key] = []
    for m in df_x['SMILES']:
        mol = Chem.MolFromSmiles(m)
        res = GetMoreauBrotoAutoofMol(mol)
        for key in labels:
            r[key].append(res[key])
    return pd.DataFrame(r)