#!/usr/bin/env python3
"""

The functions in this module allow the application of any scikit-learn regressor model. 
There are separate functions to train and evaluate (separate 
the data in train and test datasets), train with all the data available, and make
forecasts. 
"""

import copy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from epigraphhub.data.preprocessing import build_lagged_features


def rolling_predictions_mult(
    model, idx_preds, X_train, X_test, targets, horizon_forecast=14, maxlag=21
):

    if len(X_test) != None:

        preds = np.empty((len(idx_preds), horizon_forecast))

    models = {}
    for T in range(1, horizon_forecast + 1):

        tgt = targets[T][: len(X_train)]

        model.fit(X_train, tgt)

        models[f"model_{T}"] = model

        if len(X_test) != None:
            pred = model.predict(X_test.loc[idx_preds])

            preds[:, (T - 1)] = pred

    # transformando preds em um array
    if len(X_test) != None:
        train_size = len(X_train)

        y = preds.flatten()

        x = pd.period_range(
            start=X_test.index[1], end=X_test.index[-1], freq="D"
        ).to_timestamp()

        x = np.array(x)

        y = np.array(y)

        target = targets[1]

        train_size = len(X_train)

        dif = len(x) - len(y)

        if dif < 0:
            y = y[: len(y) + dif]

        df_pred = pd.DataFrame()
        df_pred["target"] = target[1:]
        df_pred["date"] = x
        df_pred["predict"] = y
        df_pred["train_size"] = [train_size] * len(df_pred)

    if X_test.empty:
        df_pred = pd.DataFrame()

    return model, df_pred


def train_eval_mult_models(
    model,
    target_name,
    data,
    ini_date="2020-03-01",
    split=0.75,
    horizon_forecast=14,
    maxlag=14,
):

    """
    Function to apply a scikit regressor model given a dataset and a target column.
    This function will train multiple models, each one specilist in predict the X + n
    days, of the target column, where n is in the range (1, number of days that you
                                                         want predict).

    This function split the data in train and test dataset and returns the predictions
    made using the test dataset.
    Important:

    params model: A model compatible with .fit and .predict scikit-learn methods.
    params target_name:string. Name of the target column.
    params data: dataframe. Dataframe with features and target column.
    params split: float. Determines which percentage of the data will be used to train the model
    params horizon_forecast: int. Number of days that will be predicted
    params max_lag: int. Number of the past days that will be used to forecast the next days

    returns: DataFrame.
    """

    target = data[target_name]

    df_lag = build_lagged_features(copy.deepcopy(data), maxlag=maxlag)

    ini_date = max(
        df_lag.index[0], target.index[0], datetime.strptime(ini_date, "%Y-%m-%d")
    )

    df_lag = df_lag[ini_date:]
    target = target[ini_date:]
    target = target.dropna()
    df_lag = df_lag.dropna()

    # remove the target column and columns related with the day that we want to predict
    df_lag = df_lag.drop(data.columns, axis=1)

    # targets
    targets = {}

    for T in np.arange(1, horizon_forecast + 1, 1):
        if T == 1:
            targets[T] = target.shift(-(T - 1))
        else:
            targets[T] = target.shift(-(T - 1))[: -(T - 1)]

    X_train, X_test, y_train, y_test = train_test_split(
        df_lag, target, train_size=split, test_size=1 - split, shuffle=False
    )

    idx_preds = pd.period_range(
        start=df_lag.index[0], end=df_lag.index[-1], freq=f"{T}D"
    ).to_timestamp()

    df_pred = rolling_predictions_mult(
        model,
        idx_preds,
        X_train,
        df_lag,
        targets,
        horizon_forecast=horizon_forecast,
        maxlag=maxlag,
    )[1]

    return df_pred


def train_eval_one_model(
    model,
    target_name,
    data,
    ini_date="2020-01-01",
    split=0.75,
    horizon_forecast=14,
    maxlag=14,
):

    """
    Function to apply a scikit regressor model given a dataset and a target column.
    This function will train one model specilist in predict the X + n
    days, of the target column, where n is in the number of days that you want predict.
    In this case, to make the predictions the mode will use the data os the last n days.

    This function split the data in train and test dataset and returns the predictions
    made using the test dataset.
    Important:

    params model: A model compatible with .fit and .predict scikit-learn methods.
    params target_name:string. Name of the target column.
    params data: dataframe. Dataframe with features and target column.
    params split: float. Determines which percentage of the data will be used to train the model
    params horizon_forecast: int. Number of days that will be predicted
    params max_lag: int. Number of the past days that will be used to forecast the next days

    returns: DataFrame.
    """

    target = data[target_name]

    df_lag = build_lagged_features(copy.deepcopy(data), maxlag=maxlag)

    ini_date = max(
        df_lag.index[0], target.index[0], datetime.strptime(ini_date, "%Y-%m-%d")
    )

    df_lag = df_lag[ini_date:]
    target = target[ini_date:]
    target = target.dropna()
    df_lag = df_lag.dropna()

    # remove the target column and columns related with the day that we want to predict
    df_lag = df_lag.drop(data.columns, axis=1)

    # targets
    targets = {}

    T = horizon_forecast
    if T == 1:
        targets[T] = target.shift(-(T - 1))
    else:
        targets[T] = target.shift(-(T - 1))[: -(T - 1)]

    X_train, X_test, y_train, y_test = train_test_split(
        df_lag, target, train_size=split, test_size=1 - split, shuffle=False
    )

    idx = pd.period_range(start=df_lag.index[0], end=df_lag.index[-1], freq=f"{T}D")

    idx = idx.to_timestamp()

    # predictions
    tgt = targets[T][: len(X_train)]

    # print('tgt')
    # print(type(tgt))
    # print(tgt.shape)

    scx = MinMaxScaler()
    scy = MinMaxScaler()

    X_train = scx.fit_transform(X_train)

    tgt = scy.fit_transform(tgt.values.reshape(-1, 1))

    model.fit(X_train, tgt.ravel())

    pred = model.predict(scx.transform(df_lag.iloc[: len(targets[T])]))

    # transformando preds em um array
    train_size = len(X_train)
    point = targets[T].index[train_size]

    y = np.array(scy.inverse_transform(pred.reshape(-1, 1)))  # type: ignore

    x = pd.period_range(
        start=df_lag.index[T], end=df_lag.index[-1], freq="D"
    ).to_timestamp()

    x = np.array(x)

    target = targets[T]

    train_size = len(X_train)

    dif = len(x) - len(y)

    if dif < 0:
        y = y[: len(y) + dif]
        df_pred = pd.DataFrame()
        df_pred["target"] = target[1:]
        df_pred["date"] = x
        df_pred["predict"] = y
        df_pred["train_size"] = [train_size] * len(df_pred)

    return df_pred
