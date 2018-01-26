import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesRegressor

def change_nan_infinite(dataframe):
    """
    Replacing NaN and infinite values from the dataframe with zeros.
    :param dataframe: Dataframe containing NaN and infinite values.
    :return data: Data with no NaN or infinite values.
    """

    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = dataframe.fillna(0)

    return data


def tree_based_feature_selection(df_x, df_y, n_features):
    df_x = change_nan_infinite(df_x)
    df_y = change_nan_infinite(df_y)
    column_names = df_x.columns
    clf = ExtraTreesRegressor()
    clf = clf.fit(df_x, df_y)
    feature_importance = clf.feature_importances_
    scores_table = pd.DataFrame({'feature': column_names, 'scores': feature_importance}).sort_values(by=['scores'],
                                                                                                     ascending=False)
    scores = scores_table['scores'].tolist()
    feature_scores = scores_table['feature'].tolist()
    selected_features = feature_scores[:n_features]
    x = pd.DataFrame(df_x, columns=selected_features)

    return selected_features, x