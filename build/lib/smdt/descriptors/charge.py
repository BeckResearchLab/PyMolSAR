
from rdkit import Chem
from rdkit.Chem import rdPartialCharges as GMCharge
import pandas as pd
import numpy

iter_step = 12

import warnings
warnings.filterwarnings("ignore")

def _CalculateElementMaxPCharge(mol, AtomicNum=6):
    """
    **Internal used only**
    Most positive charge on atom with atomic number equal to n
    """
    Hmol = Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
    res = []
    for atom in Hmol.GetAtoms():
        if atom.GetAtomicNum() == AtomicNum:
            res.append(float(atom.GetProp('_GasteigerCharge')))

    if res == []:
        return 0
    else:
        return round(max(res), 3)


def _CalculateElementMaxNCharge(mol, AtomicNum=6):
    """
    **Internal used only**
    Most negative charge on atom with atomic number equal to n
    """
    Hmol = Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
    res = []
    for atom in Hmol.GetAtoms():
        if atom.GetAtomicNum() == AtomicNum:
            res.append(float(atom.GetProp('_GasteigerCharge')))
    if res == []:
        return 0
    else:
        return round(min(res), 3)


def CalculateHMaxPCharge(mol):
    """
    Most positive charge on H atoms
    """
    return _CalculateElementMaxPCharge(mol, AtomicNum=1)


def CalculateCMaxPCharge(mol):
    """
    Most positive charge on C atoms
    """
    return _CalculateElementMaxPCharge(mol, AtomicNum=6)


def CalculateNMaxPCharge(mol):
    """
    Most positive charge on N atoms
    """
    return _CalculateElementMaxPCharge(mol, AtomicNum=7)


def CalculateOMaxPCharge(mol):
    """
    Most positive charge on O atoms
    """
    return _CalculateElementMaxPCharge(mol, AtomicNum=8)


def CalculateHMaxNCharge(mol):
    """
    Most negative charge on H atoms
    """
    return _CalculateElementMaxNCharge(mol, AtomicNum=1)


def CalculateCMaxNCharge(mol):
    """
    Most negative charge on C atoms
    """
    return _CalculateElementMaxNCharge(mol, AtomicNum=6)


def CalculateNMaxNCharge(mol):
    """
    Most negative charge on N atoms
    """
    return _CalculateElementMaxNCharge(mol, AtomicNum=7)


def CalculateOMaxNCharge(mol):
    """
    Most negative charge on O atoms
    """
    return _CalculateElementMaxNCharge(mol, AtomicNum=8)


def CalculateAllMaxPCharge(mol):
    """
    Most positive charge on ALL atoms
    """
    Hmol = Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
    res = []
    for atom in Hmol.GetAtoms():
        res.append(float(atom.GetProp('_GasteigerCharge')))
    if res == []:
        return 0
    else:
        return round(max(res), 3)


def CalculateAllMaxNCharge(mol):
    """
    Most negative charge on all atoms
    """
    Hmol = Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
    res = []
    for atom in Hmol.GetAtoms():
        res.append(float(atom.GetProp('_GasteigerCharge')))
    if res == []:
        return 0
    else:
        return round(min(res), 3)


def _CalculateElementSumSquareCharge(mol, AtomicNum=6):
    """
    **Internal used only**
    Ths sum of square Charges on all atoms with atomicnumber equal to n
    """
    Hmol = Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
    res = []
    for atom in Hmol.GetAtoms():
        if atom.GetAtomicNum() == AtomicNum:
            res.append(float(atom.GetProp('_GasteigerCharge')))
    if res == []:
        return 0
    else:
        return round(sum(numpy.square(res)), 3)


def CalculateHSumSquareCharge(mol):
    """
    The sum of square charges on all H atoms
    """
    return _CalculateElementSumSquareCharge(mol, AtomicNum=1)


def CalculateCSumSquareCharge(mol):
    """
    The sum of square charges on all C atoms
    """
    return _CalculateElementSumSquareCharge(mol, AtomicNum=6)


def CalculateNSumSquareCharge(mol):
    """
    The sum of square charges on all N atoms
    """
    return _CalculateElementSumSquareCharge(mol, AtomicNum=7)


def CalculateOSumSquareCharge(mol):
    """
    The sum of square charges on all O atoms
    """
    return _CalculateElementSumSquareCharge(mol, AtomicNum=8)


def CalculateAllSumSquareCharge(mol):
    """
    The sum of square charges on all atoms
    """
    Hmol = Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
    res = []
    for atom in Hmol.GetAtoms():
        res.append(float(atom.GetProp('_GasteigerCharge')))

    if res == []:
        return 0
    else:
        return round(sum(numpy.square(res)), 3)


def CalculateTotalPCharge(mol):
    """
    The total postive charge
    """
    Hmol = Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
    res = []
    for atom in Hmol.GetAtoms():
        res.append(float(atom.GetProp('_GasteigerCharge')))

    if res == []:
        return 0
    else:
        cc = numpy.array(res, 'd')
        return round(sum(cc[cc > 0]), 3)


def CalculateMeanPCharge(mol):
    """
    The average postive charge
    """
    Hmol = Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
    res = []
    for atom in Hmol.GetAtoms():
        res.append(float(atom.GetProp('_GasteigerCharge')))

    if res == []:
        return 0
    else:
        cc = numpy.array(res, 'd')
        return round(numpy.mean(cc[cc > 0]), 3)


def CalculateTotalNCharge(mol):
    """
    The total negative charge
    """
    Hmol = Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
    res = []
    for atom in Hmol.GetAtoms():
        res.append(float(atom.GetProp('_GasteigerCharge')))

    if res == []:
        return 0
    else:
        cc = numpy.array(res, 'd')
        return round(sum(cc[cc < 0]), 3)


def CalculateMeanNCharge(mol):
    """
    The average negative charge
    """
    Hmol = Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
    res = []
    for atom in Hmol.GetAtoms():
        res.append(float(atom.GetProp('_GasteigerCharge')))

    if res == []:
        return 0
    else:
        cc = numpy.array(res, 'd')
        return round(numpy.mean(cc[cc < 0]), 3)


def CalculateTotalAbsoulteCharge(mol):
    """
    The total absolute charge
    """
    Hmol = Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
    res = []
    for atom in Hmol.GetAtoms():
        res.append(float(atom.GetProp('_GasteigerCharge')))

    if res == []:
        return 0
    else:
        cc = numpy.array(res, 'd')
        return round(sum(numpy.absolute(cc)), 3)


def CalculateMeanAbsoulteCharge(mol):
    """
    The average absolute charge
    """
    Hmol = Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
    res = []
    for atom in Hmol.GetAtoms():
        res.append(float(atom.GetProp('_GasteigerCharge')))

    if res == []:
        return 0
    else:
        cc = numpy.array(res, 'd')
        return round(numpy.mean(numpy.absolute(cc)), 3)


def CalculateRelativePCharge(mol):
    """
    The partial charge of the most positive atom divided by
    the total positive charge.
    """
    Hmol = Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
    res = []
    for atom in Hmol.GetAtoms():
        res.append(float(atom.GetProp('_GasteigerCharge')))

    if res == []:
        return 0
    else:
        cc = numpy.array(res, 'd')
        if sum(cc[cc > 0]) == 0:
            return 0
        else:
            return round(max(res) / sum(cc[cc > 0]), 3)


def CalculateRelativeNCharge(mol):
    """
    The partial charge of the most negative atom divided
    by the total negative charge.
    """
    Hmol = Chem.AddHs(mol)
    GMCharge.ComputeGasteigerCharges(Hmol, iter_step)
    res = []
    for atom in Hmol.GetAtoms():
        res.append(float(atom.GetProp('_GasteigerCharge')))

    if res == []:
        return 0
    else:
        cc = numpy.array(res, 'd')
        if sum(cc[cc < 0]) == 0:
            return 0
        else:
            return round(min(res) / sum(cc[cc < 0]), 3)


def CalculateLocalDipoleIndex(mol):
    """
    Calculation of local dipole index (D)
    """

    GMCharge.ComputeGasteigerCharges(mol, iter_step)
    res = []
    for atom in mol.GetAtoms():
        res.append(float(atom.GetProp('_GasteigerCharge')))
    cc = [numpy.absolute(res[x.GetBeginAtom().GetIdx()] - res[x.GetEndAtom().GetIdx()]) for x in mol.GetBonds()]
    B = len(mol.GetBonds())

    return round(sum(cc) / B, 3)


def CalculateSubmolPolarityPara(mol):
    """
    Calculation of submolecular polarity parameter(SPP)
    """

    return round(CalculateAllMaxPCharge(mol) - CalculateAllMaxNCharge(mol), 3)


_Charge = {'SPP': CalculateSubmolPolarityPara,
           'LDI': CalculateLocalDipoleIndex,
           'Rnc': CalculateRelativeNCharge,
           'Rpc': CalculateRelativePCharge,
           'Mac': CalculateMeanAbsoulteCharge,
           'Tac': CalculateTotalAbsoulteCharge,
           'Mnc': CalculateMeanNCharge,
           'Tnc': CalculateTotalNCharge,
           'Mpc': CalculateMeanPCharge,
           'Tpc': CalculateTotalPCharge,
           'Qass': CalculateAllSumSquareCharge,
           'QOss': CalculateOSumSquareCharge,
           'QNss': CalculateNSumSquareCharge,
           'QCss': CalculateCSumSquareCharge,
           'QHss': CalculateHSumSquareCharge,
           'Qmin': CalculateAllMaxNCharge,
           'Qmax': CalculateAllMaxPCharge,
           'QOmin': CalculateOMaxNCharge,
           'QNmin': CalculateNMaxNCharge,
           'QCmin': CalculateCMaxNCharge,
           'QHmin': CalculateHMaxNCharge,
           'QOmax': CalculateOMaxPCharge,
           'QNmax': CalculateNMaxPCharge,
           'QCmax': CalculateCMaxPCharge,
           'QHmax': CalculateHMaxPCharge,
           }


def GetChargeforMol(mol):
    """
    Get the dictionary of constitutional descriptors for given moelcule mol
    """
    result = {}
    for DesLabel in _Charge.keys():
        result[DesLabel] = _Charge[DesLabel](mol)
    return result

def getCharge(df_x):
    """
    Calculates all Charge descriptors for the dataset
        Parameters:
            df_x: pandas.DataFrame
                SMILES DataFrame
        Returns:
            charge_descriptors: pandas.DataFrame
                Charge Descriptors DataFrame
    """

    r = {}
    for key in _Charge.keys():
        r[key] = []
    for m in df_x['SMILES']:
        mol = Chem.MolFromSmiles(m)
        res = GetChargeforMol(mol)
        for key in _Charge.keys():
            r[key].append(res[key])
    charge_descriptors = pd.DataFrame(r).round(3)
    return pd.DataFrame(charge_descriptors)

