# Data Processing

# Imports
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from smdt import utils
import numpy as np
import pandas as pd
from smdt import utils
pd.options.mode.use_inf_as_null = True


def missing_value_imputation(file, missing_value_type="NaN", strategy="mean", axis=0):
    """
    Imputes placeholder missing values in data
        Parameters:
            file: {array-like, sparse matrix}
                Sample vectors which may have have missing values
            missing_value_type:  string, optional (default="NaN")
                Placeholder for missing value. If none is given, "NaN" will be used.
            strategy: string, optional (default="mean")
                Strategy for replacing missing values. It must be one of "mean", "median", or "mode". If none is given,
                "mean" is used
            axis: int, optional (default=0)
                Imputations along rows or columns. It must be one of 0 (for columns) or 1 (for rows)
        Returns:
            file: {array-like, sparse matrix}
    """
    file.replace(missing_value_type, np.nan, inplace=True)
    # Replacing None and np.nan with the given strategy
    if axis == 0:
        if strategy == "mean":
            for i in list(file.columns):
                file[i].fillna(file[i].mean(), inplace=True)
        if strategy == "median":
            for i in list(file.columns):
                file[i].fillna(file[i].median(), inplace=True)
        if strategy == "mode":
            for i in list(file.columns):
                file[i].fillna(file[i].mode(), inplace=True)

    elif axis == 1:
        if strategy == "mean":
            file = file.T.fillna(file.mean(axis=1)).T
        if strategy == "median":
            file = file.T.fillna(file.median(axis=1)).T
        if strategy == "mode":
            file = file.T.fillna(file.mode(axis=1)).T

    if file.isnull().sum().sum() == 0:
        return file


def remove_low_variance_features(file, threshold_value=0.01):
    """
    Feature selector that removes all low-variance features.
        Parameters:
            file: pandas.DataFrame
                Input Data from which to compute variances.
            threshold_value : float, optional
                Features with a training-set variance lower than this threshold will be removed.
        Returns:
            file: {array-like, sparse matrix}
                Transformed array.
    """
    descriptors, target = utils.descriptor_target_split(file)
    column_list = list(descriptors.columns)
    selector = VarianceThreshold(threshold_value)
    transformed_arrays = selector.fit_transform(descriptors)
    transformed_columns_list = [column_list[i] for i in selector.get_support(indices=True)]
    descriptors = pd.DataFrame(transformed_arrays, columns=transformed_columns_list)
    file = utils.descriptor_target_join(descriptors, target)
    return file


def remove_high_correlated_features(file, threshold_value):
    """
    Feature selector that removes all highly-correlated features.
        Parameters:
            file: pandas.DataFrame
                Input DataFrame to remove highly correlated features.
            threshold_value : float, optional
                Features with a correlation higher than this threshold will be removed.
        Returns:
            file: {array-like, sparse matrix}
                Transformed array.
    """
    return


def univariate_feature_selection(file, k_value=10, score_function="f_regression"):
    """
    Univariate feature selection works by selecting the best features based on univariate statistical tests.
    Selects features according to the k highest scores.
        Parameters:
            file: pandas.DataFrame
                Input DataFrame to perform univariate feature selection.
            k_value: int, optional, default=10
                Number of top features to select.
            score_function: string, optional, default="f_regression"
                Scoring function that return scores and pvalues. It must be one of "f_regression" or "mutual_info_regression".
                If none is given, "f_regression" is used
        Returns:
            file: {array-like, sparse matrix}
                Transformed array.
    """
    if score_function == "f_regression":
        from sklearn.feature_selection import f_regression
        selector = SelectKBest(f_regression,k_value)
    elif score_function == "mutual_info_regression":
        from sklearn.feature_selection import mutual_info_regression
        selector = SelectKBest(mutual_info_regression, k_value)
    descriptors, target = utils.descriptor_target_split(file)
    column_list = list(descriptors.columns)
    transformed_arrays = selector.fit_transform(descriptors, target)
    transformed_columns_list = [column_list[i] for i in selector.get_support(indices=True)]
    file = pd.DataFrame(transformed_arrays, columns=transformed_columns_list)
    file = utils.descriptor_target_join(file, target)
    return file


def tree_based_feature_selection(file, n_estimators_value=10, max_features_value=None, threshold_value="mean"):
    """
    Feature selection using a tree-based estimator to compute feature importances, which in turn can be used
    to discard irrelevant features
        Parameters:
            file: pandas.DataFrame
                Input DataFrame to perform tree based feature selection.
            n_estimators_value: int, optional, default=10
                Number of trees in the forest.
            max_features_value: {int, float, string}, optional, default=None
                The number of features to consider when looking for the best split.
                If int, then consider max_features_value features at each split.
                If float, then max_features_value is a percentage and int(max_features_value*n_features) features are
                considered at each split.
                If "auto", then max_features_value=sqrt(n_features)
                If "sqrt", then max_features_value=sqrt(n_features)
                If "log2", then max_features_value=log2(n_features)
                If None, then max_features_value=n_features
            threshold_value: {int, string}, optional, default="mean"
                The threshold value to use for feature selection. Features whose importance is greater or equal are kept while
                the others are discarded. It must be one of "1.25*mean", "median", "1e-5" or "0.001". If none is given,
                "mean" is used
        Returns:
            file: {array-like, sparse matrix}
                Transformed array.
    """
    descriptors, target = utils.descriptor_target_split(file)
    column_list = list(descriptors.columns)
    clf = ExtraTreesClassifier(n_estimators=n_estimators_value, max_features=max_features_value)
    clf = clf.fit(descriptors, target)
    model = SelectFromModel(clf, prefit=True, threshold=threshold_value)
    transformed_arrays = model.transform(descriptors)
    transformed_columns_list = [column_list[i] for i in model.get_support(indices=True)]
    file = pd.DataFrame(transformed_arrays, columns=transformed_columns_list)
    file = utils.descriptor_target_join(file, target)
    return file


def rfe_feature_selection(file, step_value=1, max_features_value=3):
    """
    Select features by recursively considering smaller and smaller sets of features.
        Parameters:
            file: pandas.DataFrame
                Input DataFrame to perform RFE based feature selection.
            step_value: int, optional, default=1
                If greater than or equal to 1, then step corresponds to the (integer) number of features to remove at each
                iteration. If within (0.0, 1.0), then step corresponds to the percentage (rounded down) of features to
                remove at each iteration.
            max_features_value: int, optional, default=3
                Number of trees in the forest.
        Returns:
            file: {array-like, sparse matrix}
                Transformed array.
    """
    return

