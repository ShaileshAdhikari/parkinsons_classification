from typing import Set, Any, AnyStr, List, Tuple
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def correlation_heatmap(
        df: pd.DataFrame
) -> None:
    """
    This function draws the correlation matrix of the given dataframe.

    :param
        df: pd.DataFrame
            The dataframe whose correlation matrix is to be drawn.
    :return:
        None
    """
    # Create the correlation matrix
    corr = df.corr(method='pearson', min_periods=2).round(3)

    # Generate a mask for the upper triangle; True = do NOT show
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.tril_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(26, 10))
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,  # The data to plot
        mask=mask,  # Mask some cells
        annot=True,  # Should the values be plotted in the cells?
        vmax=1, vmin=-1, center=0,
        square=False, linewidths=.5,
        cbar_kws={"shrink": .5}  # Extra kwargs for the legend; in this case, shrink by 50%
    )


def get_feature_correlation(
        df: pd.DataFrame,
        thresh: float = 0.75,
        corr_method: AnyStr = 'pearson'
) -> pd.DataFrame:
    """
    Compute the feature correlation and sort feature pairs based on their correlation

        :param df: The dataframe with the predictor variables
        :type df: pandas.core.frame.DataFrame
        :param thresh: Threshold for correlation
        :param corr_method: Correlation compuation method
        :type corr_method: str
        :return: pandas.core.frame.DataFrame
    """

    corr_matrix_abs = df.corr(method=corr_method, min_periods=2).abs()

    mask = np.zeros_like(corr_matrix_abs, dtype=int)
    mask[np.triu_indices_from(mask)] = True

    corr_matrix_abs_us = (corr_matrix_abs * mask).replace(0, np.nan).unstack().dropna()
    sorted_correlated_features = corr_matrix_abs_us \
        .sort_values(kind="quicksort", ascending=False) \
        .reset_index()

    # Remove comparisons of the same feature
    sorted_correlated_features = sorted_correlated_features[
        (sorted_correlated_features.level_0 != sorted_correlated_features.level_1)
    ]

    # Create meaningful names for the columns
    sorted_correlated_features.columns = ['F1', 'F2', 'Corr']

    if thresh:
        return sorted_correlated_features[sorted_correlated_features['Corr'] >= thresh]

    return sorted_correlated_features


def to_remove_columns(
        df: pd.DataFrame,
        y: pd.DataFrame,
        corr_df: pd.DataFrame,
) -> List:
    """
    This function returns the list of columns to be removed from the dataframe.

    :param
        df: pd.DataFrame
            The dataframe whose columns are to be removed.
        corr_df: pd.DataFrame
            The dataframe with the correlated features.
    :return:
        List
            The list of columns to be removed.
    """
    to_remove = []
    for f1, f2 in zip(corr_df.F1, corr_df.F2):
        if df[f1].corr(y) > df[f2].corr(y):
            if f2 not in to_remove:
                to_remove.append(f2)
        else:
            if f1 not in to_remove:
                to_remove.append(f1)

    return to_remove
