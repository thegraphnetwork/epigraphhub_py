#!/usr/bin/env python3
"""
The functions in this module transform the data in a format that is
accepted by ML models (tabular data) and neural network models (3D array
data and multiple-output).
"""

from typing import Tuple, Union

import copy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def build_lagged_features(
    dt: pd.DataFrame, maxlag: int = 2, dropna: bool = True
) -> pd.DataFrame:
    """
    Builds a new DataFrame to facilitate regressing over all possible
    lagged features.

    Parameters
    ----------
    dt : pd.DataFrame
        DataFrame containing features.
    maxlag : int, optional
        Maximum lags to compute, by default 2.
    dropna : bool, optional
        If true the initial rows containing NANs due to lagging will be
        dropped, by default True.

    Returns
    -------
    pd.DataFrame
        DataFrame with the lagged values computed.
    """

    if type(dt) is pd.DataFrame:
        new_dict = {}
        for col_name in dt:
            new_dict[col_name] = dt[col_name]
            # create lagged Series
            for l in range(1, maxlag + 1):
                new_dict["%s_lag%d" % (col_name, l)] = dt[col_name].shift(l)
        res = pd.DataFrame(new_dict, index=dt.index)

    elif type(dt) is pd.Series:
        the_range = range(maxlag + 1)
        res = pd.concat([dt.shift(i) for i in the_range], axis=1)
        res.columns = ["lag_%d" % i for i in the_range]
    else:
        print("Only works for DataFrame or Series")
        return None
    if dropna:
        return res.dropna()
    else:
        return res


def preprocess_data(
    data: pd.DataFrame,
    maxlag: int,
    ini_date: Union[str, None] = None,
    end_date: Union[str, None] = None,
) -> pd.DataFrame:
    """
    This function creates a DataFrame with lagged columns that allow the
    application of ML regression model.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with datetime index and the target and features in the
        columns.
    maxlag : int
        The max number of days used to compute the lagged columns.
    ini_date : str, optional
        Determine the first day of the output dataset.
    end_date : str, optional
        Determine the last day of the output dataset.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the lagged columns.
    """

    df_lag = build_lagged_features(copy.deepcopy(data), maxlag=maxlag)

    if ini_date != None:
        df_lag = df_lag[ini_date:]

    if end_date != None:
        df_lag = df_lag[:end_date]

    df_lag = df_lag.dropna()

    return df_lag


def get_targets(target: pd.Series, predict_n: int) -> dict:
    """
    Function to create a dictionary with the targets that it will be
    used to train the ngboost model.

    Parameters
    ----------
    target : pd.Series
        Array with the values used as target.
    predict_n : int
        Number of days that it will be predicted.

    Returns
    -------
    dict
        A dictionary with the targets used to train the model.
    """

    targets = {}

    for d in range(1, predict_n + 1):
        targets[d] = target.shift(-(d))[:-(d)]

    return targets


def get_next_n_days(ini_date: str, next_days: int) -> list:
    """
    Return a list of dates with the {next_days} days after ini_date.
    This function was designed to generate the dates of the forecast
    models.

    Parameters
    ----------
    ini_date : str
        Initial date.
    next_days : int
        Number of days to be included in the list after the date in
        ini_date.

    Returns
    -------
    list
        A list with the dates computed.
    """

    next_dates = []

    a = datetime.strptime(ini_date, "%Y-%m-%d")

    for i in np.arange(1, next_days + 1):
        d_i = datetime.strftime(a + timedelta(days=int(i)), "%Y-%m-%d")

        next_dates.append(datetime.strptime(d_i, "%Y-%m-%d"))

    return next_dates


def lstm_split_data(
    df: pd.DataFrame,
    look_back: int = 12,
    ratio: float = 0.8,
    predict_n: int = 5,
    Y_column: int = 0,
) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Split the data into training and test sets. Keras expects the input
    tensor to have a shape of (nb_samples, look_back, features), and a
    output shape of (,predict_n).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    look_back : int, optional
        Number of weeks to look back before predicting. By default 12.
    ratio : float, optional
        Fraction of total samples to use for training. By default 0.8.
    predict_n : int, optional
        Number of weeks to predict. By default 5.
    Y_column : int, optional
        Column to predict. By default 0.

    Returns
    -------
    Tuple[np.array,np.array,np.array,np.array]
        X_train: array of features to train the model.
        y_train: array of targets to train the model.
        X_test: array of features to test the model.
        y_test: array of targets to test the model.
    """

    df = np.nan_to_num(df.values).astype("float64")
    # n_ts is the number of training samples also number of training sets
    # since windows have an overlap of n-1
    n_ts = df.shape[0] - look_back - predict_n + 1
    # data = np.empty((n_ts, look_back + predict_n, df.shape[1]))
    data = np.empty((n_ts, look_back + predict_n, df.shape[1]))
    for i in range(n_ts):  # - predict_):
        #         print(i, df[i: look_back+i+predict_n,0])
        data[i, :, :] = df[i : look_back + i + predict_n, :]
    # train_size = int(n_ts * ratio)
    train_size = int(df.shape[0] * ratio) - look_back - predict_n + 1
    # print(train_size)

    # We are predicting only column 0
    X_train = data[:train_size, :look_back, :]
    Y_train = data[:train_size, look_back:, Y_column]
    X_test = data[train_size:, :look_back, :]
    Y_test = data[train_size:, look_back:, Y_column]

    return X_train, Y_train, X_test, Y_test


def normalize_data(
    df: pd.DataFrame, log_transform: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Normalize features in the df table and return the normalized table
    and the values used to compute the normalization.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be normalized by the maximum value.
    log_transform : bool, optional
        If true the log transformation is applied in the data, by
        default False.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        pd.DataFrame: normalized DataFrame.
        pd.Series: Series of the max values used in the normalization.
    """

    df.fillna(0, inplace=True)
    norm = normalize(df, norm="max", axis=0)
    if log_transform == True:
        norm = np.log(norm)
    df_norm = pd.DataFrame(norm, columns=df.columns)

    return df_norm, df.max(axis=0)
