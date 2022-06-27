import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def pca_dataframe(
        df: pd.DataFrame,
        prob: float
) -> pd.DataFrame:
    """
    This function takes dataframe and probability ( accept the pca components
    which is greater than given probability (in percentage)) as arguments and return the dataframe.

    :param
        df: Dataframe
        prob: float less than 1

    :return
        DataFrame
    """
    pca_column = []
    df2 = df.copy()
    components = len(df.columns)
    # Fitting PCA model
    pca = PCA(n_components=components)
    pca.fit(np.array(df))

    variance = pca.explained_variance_ratio_
    bool_arr = variance * 100 > prob
    shorten_pca_components = np.sum(bool_arr)
    for idx in range(shorten_pca_components):
        pca_column.append(f"PCA{idx + 1}")
    # fit again in shorten pca components

    pca_new = PCA(shorten_pca_components).fit(np.array(df2))

    check = pca_new.transform(df2)
    df_new = pd.DataFrame(check, columns=pca_column)
    return df_new


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


