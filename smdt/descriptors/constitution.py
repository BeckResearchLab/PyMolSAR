from rdkit import Chem
from rdkit.Chem import Lipinski as LPK
import pandas as pd


def CalculateMolWeight(mol):
    """
    Calculation of molecular weight. Note that not including H
        Parameters:
            mol: rdkit molecule
        Returns:
            MolWeight: Molecular weight
    """
    MolWeight = 0
    for atom in mol.GetAtoms():
        MolWeight = MolWeight + atom.GetMass()

    return MolWeight


def CalculateAverageMolWeight(mol):
    """
    Calculation of average molecular weight. Note that not including H
        Parameters:
            mol: rdkit molecule
        Returns:
            AvgMolWeight: Average Molecular weight
    """
    MolWeight = 0
    for atom in mol.GetAtoms():
        MolWeight = MolWeight + atom.GetMass()
    return MolWeight / mol.GetNumAtoms()


def CalculateHydrogenNumber(mol):
    """
    Calculation of Number of Hydrogen in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            HydrogenNumber
    """
    i = 0
    Hmol = Chem.AddHs(mol)
    for atom in Hmol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            i = i + 1
    return i


def CalculateHalogenNumber(mol):
    """
    Calculation of Halogen counts in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            HalogenNumber
    """
    i = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 9 or atom.GetAtomicNum() == 17 or atom.GetAtomicNum() == 35 or atom.GetAtomicNum() == 53:
            i = i + 1
    return i


def CalculateHeteroNumber(mol):
    """
    Calculation of Hetero counts in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            HeteroNumber
    """
    i = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6 or atom.GetAtomicNum() == 1:
            i = i + 1
    return mol.GetNumAtoms() - i


def CalculateHeavyAtomNumber(mol):
    """
    Calculation of Heavy atom counts in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Heavy Atom Number
    """
    return mol.GetNumHeavyAtoms()


def _CalculateElementNumber(mol, AtomicNumber=6):
    """
    **Internal used only**
    Calculation of element counts with atomic number equal to n in a molecule
    """
    i = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == AtomicNumber:
            i = i + 1
    return i


def CalculateFlorineNumber(mol):
    """
    Calculation of Florine count in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Florine Number
    """
    return _CalculateElementNumber(mol, AtomicNumber=9)


def CalculateChlorineNumber(mol):
    """
    Calculation of Chlorine count in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Chlorine Number
    """

    return _CalculateElementNumber(mol, AtomicNumber=17)


def CalculateBromineNumber(mol):
    """
    Calculation of Bromine counts in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Bromine Number
    """
    return _CalculateElementNumber(mol, AtomicNumber=35)


def CalculateIodineNumber(mol):
    """
    Calculation of Iodine counts in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Iodine Number
    """
    return _CalculateElementNumber(mol, AtomicNumber=53)


def CalculateCarbonNumber(mol):
    """
    Calculation of Carbon number in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Carbon Number
    """
    return _CalculateElementNumber(mol, AtomicNumber=6)


def CalculatePhosphorNumber(mol):
    """
    Calculation of Phosphorus number in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Heavy Atom Number
    """
    return _CalculateElementNumber(mol, AtomicNumber=15)


def CalculateSulfurNumber(mol):
    """
    Calculation of Sulfur count in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Sulfur Number
    """
    return _CalculateElementNumber(mol, AtomicNumber=16)


def CalculateOxygenNumber(mol):
    """
    Calculation of Oxygen count in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Oxygen Number
    """
    return _CalculateElementNumber(mol, AtomicNumber=8)


def CalculateNitrogenNumber(mol):
    """
    Calculation of Nitrogen count in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Nitrogen Number
    """
    return _CalculateElementNumber(mol, AtomicNumber=7)


def CalculateRingNumber(mol):
    """
    Calculation of ring counts in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Ring Number
    """
    return Chem.GetSSSR(mol)


def CalculateRotationBondNumber(mol):
    """
    Calculation of rotation bonds count in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Rotation Bond Number
    """
    return LPK.NumRotatableBonds(mol)


def CalculateHdonorNumber(mol):
    """
    Calculation of Hydrongen bond donor count in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Hdonor Number
    """
    return LPK.NumHDonors(mol)


def CalculateHacceptorNumber(mol):
    """
    Calculation of Hydrogen bond acceptor count in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Hacceptor Number
    """
    return LPK.NumHAcceptors(mol)


def CalculateSingleBondNumber(mol):
    """
    Calculation of single bond counts in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Single Bond Number
    """
    i = 0;
    for bond in mol.GetBonds():
        if bond.GetBondType().name == 'SINGLE':
            i = i + 1
    return i


def CalculateDoubleBondNumber(mol):
    """
    Calculation of double bond counts in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Double Bond Number
    """
    i = 0;
    for bond in mol.GetBonds():
        if bond.GetBondType().name == 'DOUBLE':
            i = i + 1
    return i


def CalculateTripleBondNumber(mol):
    """
    Calculation of triple bond counts in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Triple Bond Number
    """
    i = 0;
    for bond in mol.GetBonds():
        if bond.GetBondType().name == 'TRIPLE':
            i = i + 1
    return i


def CalculateAromaticBondNumber(mol):
    """
    Calculation of aromatic bond counts in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            Aromatic Bond Number
    """
    i = 0;
    for bond in mol.GetBonds():
        if bond.GetBondType().name == 'AROMATIC':
            i = i + 1
    return i


def CalculateAllAtomNumber(mol):
    """
    Calculation of all atom counts in a molecule
        Parameters:
            mol: rdkit molecule
        Returns:
            All Atom Count
    """
    return Chem.AddHs(mol).GetNumAtoms()


def _CalculatePathN(mol, PathLength=2):
    """
    *Internal Use Only*
    Calculation of the counts of path length N for a molecule
    """
    return len(Chem.FindAllPathsOfLengthN(mol, PathLength, useBonds=1))


def CalculatePath1(mol):
    """
    Calculation of the counts of path length 1 for a molecule
    """
    return _CalculatePathN(mol, 1)


def CalculatePath2(mol):
    """
    Calculation of the counts of path length 2 for a molecule
    """
    return _CalculatePathN(mol, 2)


def CalculatePath3(mol):
    """
    Calculation of the counts of path length 3 for a molecule
    """
    return _CalculatePathN(mol, 3)


def CalculatePath4(mol):
    """
    Calculation of the counts of path length 4 for a molecule
    """
    return _CalculatePathN(mol, 4)


def CalculatePath5(mol):
    """
    Calculation of the counts of path length 5 for a molecule
    """
    return _CalculatePathN(mol, 5)


def CalculatePath6(mol):
    """
    Calculation of the counts of path length 6 for a molecule
    """
    return _CalculatePathN(mol, 6)


_constitutional = {'Weight': CalculateMolWeight,
                   'AWeight': CalculateAverageMolWeight,
                   'nhyd': CalculateHydrogenNumber,
                   'nhal': CalculateHalogenNumber,
                   'nhet': CalculateHeteroNumber,
                   'nhev': CalculateHeavyAtomNumber,
                   'ncof': CalculateFlorineNumber,
                   'ncocl': CalculateChlorineNumber,
                   'ncobr': CalculateBromineNumber,
                   'ncoi': CalculateIodineNumber,
                   'ncarb': CalculateCarbonNumber,
                   'nphos': CalculatePhosphorNumber,
                   'nsulph': CalculateOxygenNumber,
                   'noxy': CalculateOxygenNumber,
                   'nnitro': CalculateNitrogenNumber,
                   'nring': CalculateRingNumber,
                   'nrot': CalculateRotationBondNumber,
                   'ndonr': CalculateHdonorNumber,
                   'naccr': CalculateHacceptorNumber,
                   'nsb': CalculateSingleBondNumber,
                   'ndb': CalculateDoubleBondNumber,
                   'naro': CalculateAromaticBondNumber,
                   'ntb': CalculateTripleBondNumber,
                   'nta': CalculateAllAtomNumber,
                   'PC1': CalculatePath1,
                   'PC2': CalculatePath2,
                   'PC3': CalculatePath3,
                   'PC4': CalculatePath4,
                   'PC5': CalculatePath5,
                   'PC6': CalculatePath6
                   }


def GetConstitutionalofMol(mol):
    """
    Get the dictionary of constitutional descriptors for given molecule mol
        Parameters:
            mol: rdkit molecule
        Returns:
            constitution descriptors: dict
    """
    result = {}
    for DesLabel in _constitutional.keys():
        result[DesLabel] = round(_constitutional[DesLabel](mol), 3)
    return result

def getConstitutional(df_x):
    """
    Calculates all constitutional descriptors for the dataset
        Parameters:
            df_x: pandas.DataFrame
                SMILES DataFrame
        Returns:
            constitutional_descriptors: pandas.DataFrame
                Constitutional Descriptors DataFrame
    """

    r = {}
    for key in _constitutional.keys():
        r[key] = []
    for m in df_x['SMILES']:
        mol = Chem.MolFromSmiles(m)
        res = GetConstitutionalofMol(mol)
        for key in _constitutional.keys():
            r[key].append(res[key])
    constitutional_descriptors = pd.DataFrame(r).round(3)
    return pd.DataFrame(constitutional_descriptors)