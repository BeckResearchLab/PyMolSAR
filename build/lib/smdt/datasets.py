# Utility functions

# Imports
import smdt
import pandas as pd
import os
pd.set_option.use_inf_as_na = True


def load_UspInhibition():
    """
    Import the USP Inhibiton dataset
        Parameters:
            None
        Returns:
            usp_inhibiton_dataset: pandas.DataFrame
                DataFrame containing descriptors and target data.
    """
    data_path = os.path.join(smdt.__path__[0], 'examples')
    data_path = os.path.join(data_path, 'USP-Inhibition.csv')
    data = pd.read_csv(data_path)
    if 'Unnamed: 0' in data.columns:
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
    return data


def load_MeltingPoint():
    """
    Import the Melting Points dataset
        Parameters:
            None
        Returns:
            melting_point_dataset: pandas.DataFrame
                DataFrame containing descriptors and target data.
    """
    print('References: Karthikeyan, M.; Glen, R.C.; Bender, A. General melting point prediction based on a diverse compound dataset and artificial neural networks. J. Chem. Inf. Model.; 2005; 45(3); 581-590')
    data_path = os.path.join(smdt.__path__[0], 'examples')
    data_path = os.path.join(data_path, 'MeltingPoint.csv')
    data = pd.read_csv(data_path)
    if 'Unnamed: 0' in data.columns:
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
    return data


def load_LiBloodBarrier():
    """
    Import the Li Blood-Brain-Barrier Penetration dataset
        Parameters:
            None
        Returns:
            data: pandas.DataFrame
                DataFrame containing SMILES and target data.
    """
    print('Reference: \nHu Li, Chun Wei Yap, Choong Yong Ung, Ying Xue, Zhi Wei Cao and Yu Zong Chen, J. Chem. Inf. Model. 2005')
    data_path = os.path.join(smdt.__path__[0], 'examples')
    data_path = os.path.join(data_path, 'Li Blood-Brain-Barrier Penetration Set.csv')
    data = pd.read_csv(data_path)
    if 'Unnamed: 0' in data.columns:
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
    return data



def load_Sigma2ReceptorLigands():
    """
    Import the Sigma-2 Receptor Selective Ligands dataset
        Parameters:
            None
        Returns:
            data: pandas.DataFrame
                DataFrame containing SMILES and target data.
    """
    print('Reference: \nG. Nastasi, C. Miceli, V. Pittala, M.N. Modica, O. Prezzavento, G. Romeo, A. Rescifina, A. Marrazzo, E. Amata'
          'S2RSLDB: a comprehensive manually curated, internet-accessible database of the sigma-2 receptor selective ligands'
          'J. Cheminform., 9 (2017), p. 3')
    data_path = os.path.join(smdt.__path__[0], 'examples')
    data_path = os.path.join(data_path, 'Sigma-2 Receptor Selective Ligands.csv')
    data = pd.read_csv(data_path)
    if 'Unnamed: 0' in data.columns:
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
    return data
