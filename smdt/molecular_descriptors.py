
# Imports
from smdt.descriptors import topology
from smdt.descriptors import constitution
from smdt.descriptors import burden
from smdt.descriptors import basak
from smdt.descriptors import cats2d
from smdt.descriptors import charge
from smdt.descriptors import connectivity
from smdt.descriptors import estate
from smdt.descriptors import geary
from smdt.descriptors import kappa
from smdt.descriptors import moe
from smdt.descriptors import moran
from smdt.descriptors import moreaubroto
from smdt import utils
from multiprocessing import Queue
from rdkit import Chem
import pandas as pd
import math
import multiprocessing

_topology = ['W', 'AW', 'J', 'Xu', 'GMTI', 'Pol', 'DZ', 'Ipc', 'BertzCT', 'Thara', 'Tsch', 'ZM1',
             'ZM2', 'MZM1', 'MZM2', 'Qindex', 'Platt', 'diametert', 'radiust', 'petitjeant',
             'Sito', 'Hato', 'Geto', 'Arto', 'Tigdi']

_constitutional = ['Weight', 'AWeight', 'nhyd', 'nhal', 'nhet', 'nhev', 'ncof', 'ncocl', 'ncobr',
                   'ncoi', 'ncarb', 'nphos', 'nsulph', 'noxy', 'nnitro', 'nring', 'nrot', 'ndonr',
                   'naccr', 'nsb', 'ndb', 'naro', 'ntb', 'nta', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                   'PC6']

_bcut = ['bcutm16', 'bcutm15', 'bcutm14', 'bcutm13', 'bcutm12', 'bcutm11', 'bcutm10', 'bcutm9', 'bcutm8',
         'bcutm7', 'bcutm6', 'bcutm5', 'bcutm4', 'bcutm3', 'bcutm2', 'bcutm1', 'bcutv16', 'bcutv15', 'bcutv14',
         'bcutv13', 'bcutv12', 'bcutv11', 'bcutv10', 'bcutv9', 'bcutv8', 'bcutv7', 'bcutv6', 'bcutv5', 'bcutv4',
         'bcutv3', 'bcutv2', 'bcutv1', 'bcute16', 'bcute15', 'bcute14', 'bcute13', 'bcute12', 'bcute11', 'bcute10',
         'bcute9', 'bcute8', 'bcute7', 'bcute6', 'bcute5', 'bcute4', 'bcute3', 'bcute2', 'bcute1', 'bcutp16',
         'bcutp15', 'bcutp14', 'bcutp13', 'bcutp12', 'bcutp11', 'bcutp10', 'bcutp9', 'bcutp8', 'bcutp7', 'bcutp6',
         'bcutp5', 'bcutp4', 'bcutp3', 'bcutp2', 'bcutp1']

_basak = ['CIC0', 'CIC1', 'CIC2', 'CIC3', 'CIC4', 'CIC5', 'CIC6', 'SIC0', 'SIC1', 'SIC2',
          'SIC3', 'SIC4', 'SIC5', 'SIC6', 'IC0', 'IC1', 'IC2', 'IC3', 'IC4', 'IC5', 'IC6']

_cats2d = ['CATS_DD0', 'CATS_DD1', 'CATS_DD2', 'CATS_DD3', 'CATS_DD4', 'CATS_DD5', 'CATS_DD6',
           'CATS_DD7', 'CATS_DD8', 'CATS_DD9', 'CATS_DA0', 'CATS_DA1', 'CATS_DA2', 'CATS_DA3',
           'CATS_DA4', 'CATS_DA5', 'CATS_DA6', 'CATS_DA7', 'CATS_DA8', 'CATS_DA9', 'CATS_DP0',
           'CATS_DP1', 'CATS_DP2', 'CATS_DP3', 'CATS_DP4', 'CATS_DP5', 'CATS_DP6', 'CATS_DP7',
           'CATS_DP8', 'CATS_DP9', 'CATS_DN0', 'CATS_DN1', 'CATS_DN2', 'CATS_DN3', 'CATS_DN4',
           'CATS_DN5', 'CATS_DN6', 'CATS_DN7', 'CATS_DN8', 'CATS_DN9', 'CATS_DL0', 'CATS_DL1',
           'CATS_DL2', 'CATS_DL3', 'CATS_DL4', 'CATS_DL5', 'CATS_DL6', 'CATS_DL7', 'CATS_DL8',
           'CATS_DL9', 'CATS_AA0', 'CATS_AA1', 'CATS_AA2', 'CATS_AA3', 'CATS_AA4', 'CATS_AA5',
           'CATS_AA6', 'CATS_AA7', 'CATS_AA8', 'CATS_AA9', 'CATS_AP0', 'CATS_AP1', 'CATS_AP2',
           'CATS_AP3', 'CATS_AP4', 'CATS_AP5', 'CATS_AP6', 'CATS_AP7', 'CATS_AP8', 'CATS_AP9',
           'CATS_AN0', 'CATS_AN1', 'CATS_AN2', 'CATS_AN3', 'CATS_AN4', 'CATS_AN5', 'CATS_AN6',
           'CATS_AN7', 'CATS_AN8', 'CATS_AN9', 'CATS_AL0', 'CATS_AL1', 'CATS_AL2', 'CATS_AL3',
           'CATS_AL4', 'CATS_AL5', 'CATS_AL6', 'CATS_AL7', 'CATS_AL8', 'CATS_AL9', 'CATS_PP0',
           'CATS_PP1', 'CATS_PP2', 'CATS_PP3', 'CATS_PP4', 'CATS_PP5', 'CATS_PP6', 'CATS_PP7',
           'CATS_PP8', 'CATS_PP9', 'CATS_PN0', 'CATS_PN1', 'CATS_PN2', 'CATS_PN3', 'CATS_PN4',
           'CATS_PN5', 'CATS_PN6', 'CATS_PN7', 'CATS_PN8', 'CATS_PN9', 'CATS_PL0', 'CATS_PL1',
           'CATS_PL2', 'CATS_PL3', 'CATS_PL4', 'CATS_PL5', 'CATS_PL6', 'CATS_PL7', 'CATS_PL8',
           'CATS_PL9', 'CATS_NN0', 'CATS_NN1', 'CATS_NN2', 'CATS_NN3', 'CATS_NN4', 'CATS_NN5',
           'CATS_NN6', 'CATS_NN7', 'CATS_NN8', 'CATS_NN9', 'CATS_NL0', 'CATS_NL1', 'CATS_NL2',
           'CATS_NL3', 'CATS_NL4', 'CATS_NL5', 'CATS_NL6', 'CATS_NL7', 'CATS_NL8', 'CATS_NL9',
           'CATS_LL0', 'CATS_LL1', 'CATS_LL2', 'CATS_LL3', 'CATS_LL4', 'CATS_LL5', 'CATS_LL6',
           'CATS_LL7', 'CATS_LL8', 'CATS_LL9']

_charge = ['SPP', 'LDI', 'Rnc', 'Rpc', 'Mac', 'Tac', 'Mnc', 'Tnc', 'Mpc', 'Tpc', 'Qass', 'QOss',
           'QNss', 'QCss', 'QHss', 'Qmin', 'Qmax', 'QOmin', 'QNmin', 'QCmin', 'QHmin', 'QOmax',
           'QNmax', 'QCmax', 'QHmax']

_connectivity = ['Chi0', 'Chi1', 'mChi1', 'Chi2', 'Chi3', 'Chi4', 'Chi5', 'Chi6', 'Chi7', 'Chi8',
                 'Chi9', 'Chi10', 'Chi3c', 'Chi4c', 'Chi4pc', 'Chi3ch', 'Chi4ch', 'Chi5ch', 'Chi6ch',
                 'knotp', 'Chiv0', 'Chiv1', 'Chiv2', 'Chiv3', 'Chiv4', 'Chiv5', 'Chiv6', 'Chiv7',
                 'Chiv8', 'Chiv9', 'Chiv10', 'dchi0', 'dchi1', 'dchi2', 'dchi3', 'dchi4', 'Chiv3c',
                 'Chiv4c', 'Chiv4pc', 'Chiv3ch', 'Chiv4ch', 'Chiv5ch', 'Chiv6ch', 'knotpv']

_estate = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14',
           'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27',
           'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40',
           'S41', 'S42', 'S43', 'S44', 'S45', 'S46', 'S47', 'S48', 'S49', 'S50', 'S51', 'S52', 'S53',
           'S54', 'S55', 'S56', 'S57', 'S58', 'S59', 'S60', 'S61', 'S62', 'S63', 'S64', 'S65', 'S66',
           'S67', 'S68', 'S69', 'S70', 'S71', 'S72', 'S73', 'S74', 'S75', 'S76', 'S77', 'S78', 'S79',
           'Smax0', 'Smax1', 'Smax2', 'Smax3', 'Smax4', 'Smax5', 'Smax6', 'Smax7', 'Smax8', 'Smax9',
           'Smax10', 'Smax11', 'Smax12', 'Smax13', 'Smax14', 'Smax15', 'Smax16', 'Smax17', 'Smax18',
           'Smax19', 'Smax20', 'Smax21', 'Smax22', 'Smax23', 'Smax24', 'Smax25', 'Smax26', 'Smax27',
           'Smax28', 'Smax29', 'Smax30', 'Smax31', 'Smax32', 'Smax33', 'Smax34', 'Smax35', 'Smax36',
           'Smax37', 'Smax38', 'Smax39', 'Smax40', 'Smax41', 'Smax42', 'Smax43', 'Smax44', 'Smax45',
           'Smax46', 'Smax47', 'Smax48', 'Smax49', 'Smax50', 'Smax51', 'Smax52', 'Smax53', 'Smax54',
           'Smax55', 'Smax56', 'Smax57', 'Smax58', 'Smax59', 'Smax60', 'Smax61', 'Smax62', 'Smax63',
           'Smax64', 'Smax65', 'Smax66', 'Smax67', 'Smax68', 'Smax69', 'Smax70', 'Smax71', 'Smax72',
           'Smax73', 'Smax74', 'Smax75', 'Smax76', 'Smax77', 'Smax78', 'Smin0', 'Smin1', 'Smin2',
           'Smin3', 'Smin4', 'Smin5', 'Smin6', 'Smin7', 'Smin8', 'Smin9', 'Smin10', 'Smin11', 'Smin12',
           'Smin13', 'Smin14', 'Smin15', 'Smin16', 'Smin17', 'Smin18', 'Smin19', 'Smin20', 'Smin21',
           'Smin22', 'Smin23', 'Smin24', 'Smin25', 'Smin26', 'Smin27', 'Smin28', 'Smin29', 'Smin30',
           'Smin31', 'Smin32', 'Smin33', 'Smin34', 'Smin35', 'Smin36', 'Smin37', 'Smin38', 'Smin39',
           'Smin40', 'Smin41', 'Smin42', 'Smin43', 'Smin44', 'Smin45', 'Smin46', 'Smin47', 'Smin48',
           'Smin49', 'Smin50', 'Smin51', 'Smin52', 'Smin53', 'Smin54', 'Smin55', 'Smin56', 'Smin57',
           'Smin58', 'Smin59', 'Smin60', 'Smin61', 'Smin62', 'Smin63', 'Smin64', 'Smin65', 'Smin66',
           'Smin67', 'Smin68', 'Smin69', 'Smin70', 'Smin71', 'Smin72', 'Smin73', 'Smin74', 'Smin75',
           'Smin76', 'Smin77', 'Smin78']

_geary = ['GATSm1', 'GATSm2', 'GATSm3', 'GATSm4', 'GATSm5', 'GATSm6', 'GATSm7', 'GATSm8', 'GATSv1',
          'GATSv2', 'GATSv3', 'GATSv4', 'GATSv5', 'GATSv6', 'GATSv7', 'GATSv8', 'GATSe1', 'GATSe2',
          'GATSe3', 'GATSe4', 'GATSe5', 'GATSe6', 'GATSe7', 'GATSe8', 'GATSp1', 'GATSp2', 'GATSp3',
          'GATSp4', 'GATSp5', 'GATSp6', 'GATSp7', 'GATSp8']

_kappa = ['kappa1', 'kappa2', 'kappa3', 'kappam1', 'kappam2', 'kappam3', 'phi']

_moe = ['LabuteASA', 'MTPSA', 'slogPVSA0', 'slogPVSA1', 'slogPVSA2', 'slogPVSA3', 'slogPVSA4', 'slogPVSA5',
        'slogPVSA6', 'slogPVSA7', 'slogPVSA8', 'slogPVSA9', 'slogPVSA10', 'slogPVSA11', 'MRVSA0', 'MRVSA1',
        'MRVSA2', 'MRVSA3', 'MRVSA4', 'MRVSA5', 'MRVSA6', 'MRVSA7', 'MRVSA8', 'MRVSA9', 'PEOEVSA0', 'PEOEVSA1',
        'PEOEVSA2', 'PEOEVSA3', 'PEOEVSA4', 'PEOEVSA5', 'PEOEVSA6', 'PEOEVSA7', 'PEOEVSA8', 'PEOEVSA9',
        'PEOEVSA10', 'PEOEVSA11', 'PEOEVSA12', 'PEOEVSA13', 'EstateVSA0', 'EstateVSA1', 'EstateVSA2',
        'EstateVSA3', 'EstateVSA4', 'EstateVSA5', 'EstateVSA6', 'EstateVSA7', 'EstateVSA8', 'EstateVSA9',
        'EstateVSA10', 'VSAEstate0', 'VSAEstate1', 'VSAEstate2', 'VSAEstate3', 'VSAEstate4', 'VSAEstate5',
        'VSAEstate6', 'VSAEstate7', 'VSAEstate8', 'VSAEstate9', 'VSAEstate10']

_moran = ['MATSm1', 'MATSm2', 'MATSm3', 'MATSm4', 'MATSm5', 'MATSm6', 'MATSm7', 'MATSm8', 'MATSv1',
          'MATSv2', 'MATSv3', 'MATSv4', 'MATSv5', 'MATSv6', 'MATSv7', 'MATSv8', 'MATSe1', 'MATSe2',
          'MATSe3', 'MATSe4', 'MATSe5', 'MATSe6', 'MATSe7', 'MATSe8', 'MATSp1', 'MATSp2', 'MATSp3',
          'MATSp4', 'MATSp5', 'MATSp6', 'MATSp7', 'MATSp8']

_moreaubroto = ['ATSm1', 'ATSm2', 'ATSm3', 'ATSm4', 'ATSm5', 'ATSm6', 'ATSm7', 'ATSm8', 'ATSv1', 'ATSv2',
                'ATSv3', 'ATSv4', 'ATSv5', 'ATSv6', 'ATSv7', 'ATSv8', 'ATSe1', 'ATSe2', 'ATSe3', 'ATSe4',
                'ATSe5', 'ATSe6', 'ATSe7', 'ATSe8', 'ATSp1', 'ATSp2', 'ATSp3', 'ATSp4', 'ATSp5', 'ATSp6',
                'ATSp7', 'ATSp8']


def getAllDescriptorsforMol(mol):
    topology_descriptors = topology.GetTopologyofMol(mol)
    constitution_descriptors = constitution.GetConstitutionalofMol(mol)
    burden_descriptors = burden.GetBurdenofMol(mol)
    basak_descriptors = basak.GetBasakofMol(mol)
    cats2d_descriptors = cats2d.CATS2DforMol(mol)
    charge_descriptors = charge.GetChargeforMol(mol)
    connectivity_descriptors = connectivity.GetConnectivityforMol(mol)
    estate_descriptors = estate._GetEstateforMol(mol)
    gearyauto_descriptors = geary.GetGearyAutoofMol(mol)
    kappa_descriptors = kappa.GetKappaofMol(mol)
    moe_descriptors = moe.GetMOEofMol(mol)
    moran_descriptors = moran.GetMoranAutoofMol(mol)
    moreaubroto_descriptors = moreaubroto.GetMoreauBrotoAutoofMol(mol)

    descriptor_list = [topology_descriptors, constitution_descriptors, burden_descriptors, basak_descriptors,
                       cats2d_descriptors, charge_descriptors, connectivity_descriptors, estate_descriptors,
                       gearyauto_descriptors, kappa_descriptors, moe_descriptors, moran_descriptors,
                       moreaubroto_descriptors]

    final_values = []
    for i in descriptor_list:
        final_values = final_values + list(i.values())

    return final_values

descriptor_fn = {'topology': topology.GetTopologyofMol, 'constitutional': constitution.GetConstitutionalofMol,
    'burden': burden.GetBurdenofMol, 'basak': basak.GetBasakofMol, 'cats2d': cats2d.CATS2DforMol,
    'charge': charge.GetChargeforMol, 'connectivity': connectivity.GetConnectivityforMol, 'estate': estate._GetEstateforMol,
    'geary': geary.GetGearyAutoofMol, 'kappa': kappa.GetKappaofMol, 'moe': moe.GetMOEofMol, 'moran': moran.GetMoranAutoofMol,
    'moreaubroto': moreaubroto.GetMoreauBrotoAutoofMol}

descriptor_list = {'topology': _topology, 'constitutional': _constitutional, 'burden': _bcut, 'basak': _basak,
                   'cats2d': _cats2d, 'charge': _charge, 'connectivity': _connectivity, 'estate': _estate,
                   'geary': _geary, 'kappa': _kappa, 'moe': _moe, 'moran': _moran, 'moreaubroto': _moreaubroto}


def getDescriptors(data, descriptor_type = 'topology'):
    smiles, target = utils.descriptor_target_split(data)
    cols = descriptor_list[descriptor_type]
    AllDescriptors = pd.DataFrame(columns=cols)
    print('\nCalculating %s descriptors...'%descriptor_type)
    for i in range(len(smiles)):
        print('Row %d out of %d' % (i + 1, len(smiles)), end='')
        print('\r', end='')
        AllDescriptors.loc[i] = descriptor_fn[descriptor_type](Chem.MolFromSmiles(smiles['SMILES'][i]))
    final_df = utils.descriptor_target_join(AllDescriptors, target)
    print('\nCalculating %s descriptors completed.'%descriptor_type)
    return final_df


def getAllDescriptors(data):
    smiles, target = utils.descriptor_target_split(data)
    cols = _topology + _constitutional + _bcut + _basak + _cats2d + _charge + _connectivity + _estate + _geary + _kappa + _moe + _moran + _moreaubroto
    AllDescriptors = pd.DataFrame(columns=cols)
    print('\nCalculating Molecular Descriptors...')
    for i in range(len(smiles)):
        print('Row %d out of %d' % (i + 1, len(smiles)), end='')
        print('\r', end='')
        AllDescriptors.loc[i] = getAllDescriptorsforMol(Chem.MolFromSmiles(smiles['SMILES'][i]))
    final_df = utils.descriptor_target_join(AllDescriptors, target)
    print('\nCalculating Molecular Descriptors Completed.')
    return final_df