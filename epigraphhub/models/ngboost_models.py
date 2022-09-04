#!/usr/bin/env python3
"""

The functions in this module allow the application of the
ngboost regressor model. There are separate functions to train and evaluate (separate
the data in train and test datasets), train with all the data available, and make
forecasts. Also, there are functions to apply these methods in just one canton or all the cantons.

"""

import copy
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from joblib import dump, load
from ngboost import NGBRegressor
from ngboost.distns import LogNormal, Poisson
from ngboost.learners import default_tree_learner
from ngboost.scores import CRPScore, LogScore
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error as msle

from epigraphhub.data.preprocessing import build_lagged_features

params_model = {
    "Base": default_tree_learner,
    "Dist": LogNormal,
    "Score": LogScore,
    "natural_gradient": True,
    "verbose": False,
    "col_sample": 0.9,
    "n_estimators": 100,
    "learning_rate": 0.1,
    "validation_fraction": 0.15,
    "early_stopping_rounds": 10,
}


def get_targets(target, predict_n):
    """
    Function to create a dictionary that it will be used train the ngboost model
    :params target: array with the values used as target.
    :params predict_n: int. Number os days that it will be predicted.
    """

    targets = {}

    for d in range(1, predict_n + 1):
        targets[d] = target.shift(-(d))[:-(d)]

    return targets


def train_eval_ngb(
    target_name,
    data,
    ini_date="2020-03-01",
    end_train_date=None,
    end_date=None,
    ratio=0.75,
    predict_n=14,
    look_back=14,
    kwargs=params_model,
    path=None,
    name=None,
    save=False,
):

    """
    Function to apply a ngboost regressor model given a dataset and a target column.
    This function will train multiple models, each one specilist in predict the X + n
    days, of the target column, where n is in the range (1, number of days that you
                                                         want predict).
    This function split the data in train and test dataset and returns the predictions
    made using the test dataset.
    Important:

    :params target_name:string. Name of the target column.
    :params data: dataframe. Dataframe with features and target column.
    :params ini_date: string or None. Determines the beggining of the train dataset
    :params end_train_date: string or None. Determines the beggining of end of train dataset. If end_train_date
                           is not None, then ratio isn't used.
    :params end_date: string or None. Determines the end of the dataset used in validation.
    :params ratio: float. Determines which percentage of the data will be used to train the model
    :params predict_n: int. Number of days that will be predicted
    :params look_back: int. Number of the last days that will be used to forecast the next days
    :params parameters_model: dict with the params that will be used in the ngboost
                             regressor model.
    :params path: sting. It indicates where save the models trained.
    :params name: string. It indicates which name use to save the models trained.
    :params save: boolean. If True the models trained are saved.

    returns: DataFrame.
    """

    df_lag = build_lagged_features(copy.deepcopy(data), maxlag=look_back)

    if ini_date != None:
        df_lag = df_lag[ini_date:]

    if end_date != None:
        df_lag = df_lag[:end_date]

    df_lag = df_lag.dropna()

    target = df_lag[target_name]

    targets = get_targets(target, predict_n)

    # remove the target column and columns related with the day that we want to predict
    df_lag = df_lag.drop(data.columns, axis=1)

    if end_train_date == None:
        X_train = df_lag.iloc[: int(df_lag.shape[0] * ratio)]

    else:
        X_train = df_lag.loc[:end_train_date]

    idx = pd.period_range(
        start=df_lag.index[0], end=df_lag.index[-1], freq=f"{predict_n}D"
    )

    idx = idx.to_timestamp()

    preds5 = np.empty((len(idx), predict_n))
    preds50 = np.empty((len(idx), predict_n))
    preds95 = np.empty((len(idx), predict_n))

    for T in range(1, predict_n + 1):

        tgt = targets[T][: len(X_train)]

        model = NGBRegressor(**kwargs)

        model.fit(X_train, tgt)

        pred = model.pred_dist(df_lag.loc[idx], max_iter=model.best_val_loss_itr)

        pred50 = pred.median()

        pred5, pred95 = pred.interval(alpha=0.95)

        preds5[:, (T - 1)] = pred5
        preds50[:, (T - 1)] = pred50
        preds95[:, (T - 1)] = pred95

        if save:
            dump(model, f"{path}/{name}_{T}.joblib")

    train_size = len(X_train)

    y5 = preds5.flatten()
    y50 = preds50.flatten()
    y95 = preds95.flatten()

    x = pd.period_range(
        start=df_lag.index[1], end=df_lag.index[-1], freq="D"
    ).to_timestamp()

    x = np.array(x)

    y5 = np.array(y5)

    y50 = np.array(y50)

    y95 = np.array(y95)

    train_size = len(X_train)

    dif = len(x) - len(y5)

    if dif < 0:
        y5 = y5[: len(y5) + dif]
        y50 = y50[: len(y50) + dif]
        y95 = y95[: len(y95) + dif]

    df_pred = pd.DataFrame()
    df_pred["target"] = target[1:]
    df_pred["date"] = x
    df_pred["lower"] = y5
    df_pred["median"] = y50
    df_pred["upper"] = y95
    df_pred["train_size"] = [train_size] * len(df_pred)
    df_pred.set_index("date", inplace=True)
    df_pred.index = pd.to_datetime(df_pred.index)

    return df_pred


def train_ngb(
    target_name,
    data,
    ini_date="2020-03-01",
    end_date=None,
    predict_n=14,
    look_back=14,
    path="../opt/models/saved_models/ml",
    save=True,
    kwargs=params_model,
):

    """
    Function to train multiple ngboost regressor models given a dataset and a target column.
    This function will train multiple models, each one specilist in predict the X + n
    days, of the target column, where n is in the range (1, number of days that you
                                                         want predict).
    This function will train the model with all the data available and will save the model
    that will be used to make forecasts.

    params target_name:string. Name of the target column.
    params ini_date: string or None. Determines the beggining of the train dataset
    params end_date: string or None. Determines the end of the train dataset
    params predict_n: int. Number of days that will be predicted
    params look_back: int. Number of the last days that will be used to forecast the next days
    params path: string. Indicates where the models will be saved
    params save: boolean. If True the models is saved
    params kwargs: dict with the params that will be used in the ngboost
                             regressor model.

    returns: list with the {predict_n} models saved
    """

    if ini_date != None:
        data = data.loc[ini_date:]

    if end_date != None:
        data = data.loc[:end_date]

    df_lag = build_lagged_features(copy.deepcopy(data), maxlag=look_back)

    df_lag = df_lag.dropna()

    target = df_lag[target_name]

    models = []

    # remove the target column and columns related with the day that we want to predict
    df_lag = df_lag.drop(data.columns, axis=1)

    targets = get_targets(target, predict_n)

    X_train = df_lag.iloc[:-1]

    for T in range(1, predict_n + 1):
        tgt = targets[T]

        model = NGBRegressor(**kwargs)

        model.fit(X_train.iloc[: len(tgt)], tgt)

        if save:
            dump(model, f"{path}/ngboost_{target_name}_{T}D.joblib")

        models.append(model)

    return models


def forecast_ngb(
    target_name,
    data,
    end_date=None,
    predict_n=14,
    look_back=14,
    path="../opt/models/saved_models/ml",
):

    """
    Function to load multiple ngboost regressor model trained with the function
    `training_model` and make the forecast.

    Important:
    predict_n and max_lag need have the same value used in training_model
    Only the last that of the dataset will be used to forecast the next
    predict_n days.

    params target_name:string. Name of the target column.
    params data: dataframe. Dataframe with features and target column.
    params end_date: string. Determines from what day the forecast will be computed.
    params predict_n: int. Number of days that will be predicted
    params look_back: int. Number of the last days that will be used to forecast the next days
    params path: string. Indicates where the model is save to the function load the model

    returns: DataFrame.
    """

    df_lag = build_lagged_features(copy.deepcopy(data), maxlag=look_back)

    if end_date != None:
        df_lag = df_lag.loc[:end_date]

    # remove the target column and columns related with the day that we want to predict
    df_lag = df_lag.drop(data.columns, axis=1)

    forecasts5 = []
    forecasts50 = []
    forecasts95 = []

    for T in range(1, predict_n + 1):

        if os.path.exists(f"{path}/ngboost_{target_name}_{T}D.joblib"):

            model = load(f"{path}/ngboost_{target_name}_{T}D.joblib")

            forecast = model.pred_dist(df_lag.iloc[-1:])

            forecast50 = forecast.median()

            forecast5, forecast95 = forecast.interval(alpha=0.95)

            forecasts5.append(forecast5)
            forecasts50.append(forecast50)
            forecasts95.append(forecast95)

        else:
            forecasts5.append(0)
            forecasts50.append(0)
            forecasts95.append(0)

    # transformando preds em um array
    forecast_dates = []

    last_day = datetime.strftime((df_lag.index)[-1], "%Y-%m-%d")

    a = datetime.strptime(last_day, "%Y-%m-%d")

    for i in np.arange(1, predict_n + 1):

        d_i = datetime.strftime(a + timedelta(days=int(i)), "%Y-%m-%d")

        forecast_dates.append(datetime.strptime(d_i, "%Y-%m-%d"))

    df_for = pd.DataFrame()
    df_for["date"] = forecast_dates
    df_for["lower"] = np.array(forecasts5)
    df_for["median"] = np.array(forecasts50)
    df_for["upper"] = np.array(forecasts95)
    df_for.set_index("date", inplace=True)

    return df_for


def compute_metrics(df_pred):
    """
    This function evaluates the predictions obtained in the `train_eval_ngb()` function
    in the train and test sample. The predictions must be saved in a dataset with the following columns:
    'median', 'target' and 'train_size'.
    This function uses the following metrics:
    - explained variance score;
    - mean absolute error;
    - mean squared error;
    - root mean squared error;
    - mean squared log error;
    - mean absolute percentage error.

    To compute this metrics we use the implementations of the sklearn.metrics package.
    """

    metrics = [
        "explained_variance_score",
        "mean_absolute_error",
        "mean_squared_error",
        "root_mean_squared_error",
        "mean_squared_log_error",
        "mean_absolute_percentage_error",
    ]

    # computing error in train sample
    df_metrics = pd.DataFrame(columns=["metrics", "in_sample", "out_sample"])

    df_metrics["metrics"] = metrics

    split = df_pred["train_size"][0]
    y_true_in = df_pred["target"].iloc[:split]
    y_pred_in = df_pred["median"].iloc[:split]
    y_true_out = df_pred["target"].iloc[split:]
    y_pred_out = df_pred["median"].iloc[split:]

    df_metrics["in_sample"] = [
        evs(y_true_in, y_pred_in),
        mae(y_true_in, y_pred_in),
        mse(y_true_in, y_pred_in),
        mse(y_true_in, y_pred_in, squared=False),
        msle(y_true_in, y_pred_in),
        mape(y_true_in, y_pred_in),
    ]

    df_metrics["out_sample"] = [
        evs(y_true_out, y_pred_out),
        mae(y_true_out, y_pred_out),
        mse(y_true_out, y_pred_out),
        mse(y_true_out, y_pred_out, squared=False),
        msle(y_true_out, y_pred_out),
        mape(y_true_out, y_pred_out),
    ]

    df_metrics.set_index("metrics", inplace=True)

    return df_metrics
