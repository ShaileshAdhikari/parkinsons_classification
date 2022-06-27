from typing import AnyStr, List, Tuple
import pandas as pd
import numpy as np


def read_data(data_path: AnyStr) -> pd.DataFrame:
    """
    Reads data from a csv file.

    :param data_path: path to the csv file
    :return: dataframe containing the data
    """
    # Reading data from csv file
    dataframe = pd.read_csv(data_path)

    # Breaking Dataframe to individual sections
    baseline = dataframe.iloc[:, 2:23]
    intensity = dataframe.iloc[:, 23:26]
    frequency = dataframe.iloc[:, 26:30]
    bandwidth = dataframe.iloc[:, 30:34]
    mfcc = dataframe.iloc[:,57:70]
    y = dataframe['class']

    # Removing unnecessary columns
    std_value_columns = [cols for cols in baseline.columns if cols.__contains__('std')]
    other_columns = list(set(baseline.columns) - set(std_value_columns))

    # Merging all the dataframes and returning
    return baseline[other_columns].join([intensity, frequency, bandwidth, mfcc, y])


