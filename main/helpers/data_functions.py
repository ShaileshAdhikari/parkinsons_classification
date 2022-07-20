from typing import Tuple, Any
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

RANDOM_STATE = 142


# split into train and test
def get_train_test(
        df: pd.DataFrame,
        target: str = 'class',
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
    X = df.drop(target, axis=1)
    Y = df[target]
    return train_test_split(X, Y, test_size=test_size, random_state=RANDOM_STATE, stratify=Y)


# dataframe for plotting ratios of classes
def get_class_ratios(
        train: pd.DataFrame,
        test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get dataframe for plotting ratios of classes

    :param
        train: dataframe of y_train
        test: dataframe of y_test

    :return:
        dataframe for plotting ratios of classes
    """
    return pd.DataFrame({
        'Training': train.value_counts(normalize=True),
        'Testing': test.value_counts(normalize=True)
    })
