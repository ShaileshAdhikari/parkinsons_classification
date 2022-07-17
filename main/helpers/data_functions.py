from typing import Tuple, Any
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

RANDOM_STATE = 142


# split into train and test
def get_train_test(
        X: pd.DataFrame,
        Y: pd.DataFrame,
        test_size: float = 0.2
) -> Tuple[Any, Any, Any, Any]:
    """
    Split data into train and test

    :param
        X: data
        Y: labels
        test_size: size of test set

    :return:
        tuple of train and test data
    """
    return train_test_split(X, Y, test_size=test_size, random_state=RANDOM_STATE, stratify=Y)


def normalize(
        df: pd.DataFrame,
        mode: str
) -> pd.DataFrame:
    """
    This function applies different normalization in given dataframe
    and returns the normalized dataframe.

    :param
        df: Dataframe
        mode: type of normalization
    :return
        return pd.DataFrame

    """
    if mode == "standard":
        for col in df.columns:
            df[col] = (df[col] - np.mean(df[col])) / (np.std(df[col]))
    elif mode == "minmax":
        for col in df.columns:
            df[col] = (df[col] - min(df[col])) / (max(df[col]) - min(df[col]))
    else:
        raise ValueError(
            "Invalid mode, %s selected. Please select either 'standard' or 'minmax' "
            % mode)
    return df
