# Utility functions

# Imports
from sklearn.model_selection import train_test_split
import smdt
import numpy as np
import pandas as pd
import os
pd.options.mode.use_inf_as_null = True


def test_train_split(file, test_size_value=0.25, train_size_value=None):
    """
    Split Input DataFrame into Training and Testing Data
        Parameters:
            file: pandas.DataFrame
                Input DataFrame containing descriptors and target data.
            test_size_value : float, int, default=0.25
                If float, should be between 0.0 and 1.0 and represent the proportion of the data to include in the
                test split. If int, represents the absolute number of test samples. If None, the value is set to the
                complement of the train size.
            train_size_value : float, int, or None, default None
                If float, should be between 0.0 and 1.0 and represent the proportion of the data to include in the
                train split. If int, represents the absolute number of train samples. If None, the value is automatically
                set to the complement of the test size.
        Returns:
            train: pandas.DataFrame
                DataFrame containing training data.
            test: pandas.DataFrame
                DataFrame containing testing data.
    """
    descriptors, target = descriptor_target_split(file)
    x_train, x_test, y_train, y_test = train_test_split(descriptors, target, test_size=test_size_value,
                                                        train_size=train_size_value)
    train = descriptor_target_join(x_train, y_train)
    train.reset_index(inplace=True)
    train.drop('index', axis=1, inplace=True)
    test = descriptor_target_join(x_test, y_test)
    test.reset_index(inplace=True)
    test.drop('index', axis=1, inplace=True)
    return train, test


def descriptor_target_split(file):
    """
    Split the input data into descriptors and target DataFrames
        Parameters:
            file: pandas.DataFrame
                Input DataFrame containing descriptors and target data.
        Returns:
            descriptors: pandas.DataFrame
                Descriptors DataFrame.
            target: pandas.DataFrame
                Target DataFrame.
    """
    target = file.loc[:, file.columns == 'Target']
    descriptors = file.loc[:, file.columns != 'Target']
    return descriptors, target


def descriptor_target_join(descriptors, target):
    """
    Merge the Descriptors and Target DataFrames
        Parameters:
            descriptors: pandas.DataFrame
                Descriptors DataFrame.
            target: pandas.DataFrame
                Target DataFrame.
        Returns:
            file: pandas.DataFrame
                Input DataFrame containing descriptors and target data.
    """

    descriptors['Target'] = target['Target']
    file = descriptors
    return file

