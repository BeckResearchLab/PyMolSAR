import numpy as np
import pandas as pd

def change_nan_infinite(dataframe):
    """
    Replacing NaN and infinite values from the dataframe with zeros.
    :param dataframe: Dataframe containing NaN and infinite values.
    :return data: Data with no NaN or infinite values.
    """

    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = dataframe.fillna(0)

    return data