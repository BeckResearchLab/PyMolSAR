"""
##############################################################################
This module is to compute the estate fingerprints and values based on Kier
and Hall's paper.
"""

from rdkit.Chem.EState import Fingerprinter  as ESFP
from rdkit import Chem
import pandas as pd
import AtomTypes as ATEstate
import numpy

def _CalculateEState(mol, skipH=1):
    """
    #################################################################
    **Internal used only**

    Get the EState value of each atom in a molecule
    #################################################################
    """
    mol = Chem.AddHs(mol)
    if skipH == 1:
        mol = Chem.RemoveHs(mol)
    tb1 = Chem.GetPeriodicTable()
    nAtoms = mol.GetNumAtoms()
    Is = numpy.zeros(nAtoms, numpy.float)
    for i in range(nAtoms):
        at = mol.GetAtomWithIdx(i)
        atNum = at.GetAtomicNum()
        d = at.GetDegree()
        if d > 0:
            h = at.GetTotalNumHs()
            dv = tb1.GetNOuterElecs(atNum) - h
            # dv=numpy.array(_AtomHKDeltas(at),'d')
            N = _GetPrincipleQuantumNumber(atNum)
            Is[i] = (4.0 / (N * N) * dv + 1) / d
    dists = Chem.GetDistanceMatrix(mol, useBO=0, useAtomWts=0)
    dists += 1
    accum = numpy.zeros(nAtoms, numpy.float)
    for i in range(nAtoms):
        for j in range(i + 1, nAtoms):
            p = dists[i, j]
            if p < 1e6:
                temp = (Is[i] - Is[j]) / (p * p)
                accum[i] += temp
                accum[j] -= temp
    res = accum + Is
    return res


def _GetPrincipleQuantumNumber(atNum):
    """
    #################################################################
    *Internal Use Only*

    Get the principle quantum number of atom with atomic

    number equal to atNum
    #################################################################
    """
    if atNum <= 2:
        return 1
    elif atNum <= 10:
        return 2
    elif atNum <= 18:
        return 3
    elif atNum <= 36:
        return 4
    elif atNum <= 54:
        return 5
    elif atNum <= 86:
        return 6
    else:
        return 7


def CalculateEstateFingerprint(mol):
    """
    #################################################################
    The Calculation of EState Fingerprints.

    It is the number of times each possible atom type is hit.

    Usage:

        result=CalculateEstateFingerprint(mol)

        Input: mol is a molecule object.

        Output: result is a dict form containing 79 estate fragments.
    #################################################################
    """
    temp = ESFP.FingerprintMol(mol)
    res = {}
    for i, j in enumerate(temp[0]):
        res['Sfinger' + str(i + 1)] = j

    return res


def CalculateEstateValue(mol):
    """
    #################################################################
    The Calculate of EState Values.

    It is the sum of the Estate indices for atoms of each type.

    Usage:

        result=CalculateEstateValue(mol)

        Input: mol is a molecule object.

        Output: result is a dict form containing 79 estate values.
    #################################################################
    """
    temp = ESFP.FingerprintMol(mol)
    res = {}
    for i, j in enumerate(temp[1]):
        res['S' + str(i + 1)] = round(j, 3)

    return res


def CalculateMaxAtomTypeEState(mol):
    """
    #################################################################
    Calculation of maximum of E-State value of specified atom type

    res---->dict type

    Usage:

        result=CalculateMaxAtomTypeEState(mol)

        Input: mol is a molecule object.

        Output: result is a dict form containing 79 max estate values.
    #################################################################
    """
    AT = ATEstate.GetAtomLabel(mol)
    Estate = _CalculateEState(mol)
    res = []
    for i in AT:
        if i == []:
            res.append(0)
        else:
            res.append(max([Estate[k] for k in i]))
    ESresult = {}
    for n, es in enumerate(res):
        ESresult['Smax' + str(n)] = round(es, 3)

    return ESresult


def CalculateMinAtomTypeEState(mol):
    """
    #################################################################
    Calculation of minimum of E-State value of specified atom type

    res---->dict type

    Usage:

        result=CalculateMinAtomTypeEState(mol)

        Input: mol is a molecule object.

        Output: result is a dict form containing 79 min estate values.
    #################################################################
    """
    AT = ATEstate.GetAtomLabel(mol)
    Estate = _CalculateEState(mol)
    res = []
    for i in AT:
        if i == []:
            res.append(0)
        else:
            res.append(min([Estate[k] for k in i]))
    ESresult = {}
    for n, es in enumerate(res):
        ESresult['Smin' + str(n)] = round(es, 3)

    return ESresult


def GetEstateforMol(mol):
    """
    #################################################################
    Obtain all descriptors related to Estate.
    Usage:

        result=GetEstate(mol)

        Input: mol is a molecule object.

        Output: result is a dict form containing all estate values.
    #################################################################
    """
    result = {}
    result.update(CalculateEstateFingerprint(mol))
    result.update(CalculateEstateValue(mol))
    result.update(CalculateMaxAtomTypeEState(mol))
    result.update(CalculateMinAtomTypeEState(mol))

    return result


def _GetEstateforMol(mol):
    """
    #################################################################
    Obtain all Estate descriptors except Estate fingerprints .
    Usage:

        result=_GetEstate(mol)

        Input: mol is a molecule object.

        Output: result is a dict form containing all estate values.
    #################################################################
    """
    result = {}
    result.update(CalculateEstateValue(mol))
    result.update(CalculateMaxAtomTypeEState(mol))
    result.update(CalculateMinAtomTypeEState(mol))

    return result

labels = []
for i in range(1,79):
    labels.append('S' + str(i))
    labels.append('Smax' + str(i))
    labels.append('Smin' + str(i))
labels.append('S79')

def GetEstate(df_x):
    r = {}
    for key in labels:
        r[key] = []
    for m in df_x['SMILES']:
        mol = Chem.MolFromSmiles(m)
        res = _GetEstateforMol(mol)
        for key in labels:
            r[key].append(res[key])
    return pd.DataFrame(r)