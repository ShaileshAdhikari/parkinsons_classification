from typing import Set, Any
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def draw_correlation_matrix(
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


def correlation(
        dataset: pd.DataFrame,
        threshold: float = 0.9,
) -> Set[Any]:

    """
    This function selects highly correlated features.
    :param
        dataset: pd.DataFrame
            The dataframe whose correlation matrix is to be drawn.
        threshold: float
            The threshold value for the correlation.
    :return:
        Set of Correlated features.
    """
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr(method='pearson', min_periods=2)
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
