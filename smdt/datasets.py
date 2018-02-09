# Utility functions

# Imports
import smdt
import pandas as pd
import os
pd.options.mode.use_inf_as_null = True


def usp_inhibition():
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


def melting_point():
    """
    Import the Melting Points dataset
        Parameters:
            None
        Returns:
            melting_point_dataset: pandas.DataFrame
                DataFrame containing descriptors and target data.
    """
    data_path = os.path.join(smdt.__path__[0], 'examples')
    data_path = os.path.join(data_path, 'MeltingPoint.csv')
    data = pd.read_csv(data_path)
    if 'Unnamed: 0' in data.columns:
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
    return data
