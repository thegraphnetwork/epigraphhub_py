#!/usr/bin/env python3
"""
The functions in this module allow the application of the 
ngboost regressor model for time series. There are separate functions to train and evaluate (separate 
the data in train and test datasets), train with all the data available, and make
forecasts. Also, there are functions to apply these methods in just one canton or all
the cantons of switzerland. 
"""

import copy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from joblib import dump, load
from ngboost import NGBRegressor
from ngboost.distns import LogNormal
from ngboost.learners import default_tree_learner
from ngboost.scores import LogScore
from sklearn.model_selection import train_test_split

from epigraphhub.data.get_data import get_cluster_data
from epigraphhub.data.preprocessing import build_lagged_features

params_model = {
    "Base": default_tree_learner,
    "Dist": LogNormal,
    "Score": LogScore,
    "natural_gradient": True,
    "verbose": False,
    "col_sample": 0.9,
    "n_estimators": 30,
    "learning_rate": 0.1,
    "validation_fraction": 0.15,
    "early_stopping_rounds": 50,
}


def rolling_predictions(
    target_name,
    data,
    ini_date="2020-03-01",
    split=0.75,
    horizon_forecast=14,
    maxlag=14,
    kwargs=params_model,
):

    """
    Function to apply a ngboost regressor model given a dataset and a target column.
    This function will train multiple models, each one specilist in predict the X + n
    days, of the target column, where n is in the range [1, number of days that you
                                                         want predict].
    This function split the data in train and test dataset and returns the predictions
    made using the test dataset. The `data` dataframe must have a datetime index.

    Important:

    :params target_name:string. Name of the target column.
    :params data: dataframe. Dataframe with features and target column.
    :params ini_date: string. Determines the beggining of the train dataset
    :params split: float. Determines which percentage of the data will be used to train the model
    :params horizon_forecast: int. Number of days that will be predicted
    :params max_lag: int. Number of the last days that will be used to forecast the next days
    :params parameters_model: dict with the params that will be used in the ngboost
                             regressor model.

    returns: DataFrame, list of the models trained, X_train (features used training the model), targets
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

    models = {}

    targets = {}

    for T in np.arange(1, horizon_forecast + 1, 1):
        targets[T] = target.shift(-(T))[:-(T)]

    X_train, X_test, y_train, y_test = train_test_split(
        df_lag, target, train_size=split, test_size=1 - split, shuffle=False
    )

    if np.sum(target) > 0.0:

        idx = pd.period_range(
            start=df_lag.index[0], end=df_lag.index[-1], freq=f"{horizon_forecast}D"
        )

        idx = idx.to_timestamp()

        preds5 = np.empty((len(idx), horizon_forecast))
        preds50 = np.empty((len(idx), horizon_forecast))
        preds95 = np.empty((len(idx), horizon_forecast))

        for T in range(1, horizon_forecast + 1):

            tgt = targets[T][: len(X_train)]

            i = 0

            while i < len(tgt):
                if tgt[i] <= 0:

                    tgt[i] = 0.01
                i = i + 1

            model = NGBRegressor(**kwargs)

            models[T] = model.fit(X_train, tgt)

            pred = model.pred_dist(df_lag.loc[idx])

            pred50 = pred.median()

            pred5, pred95 = pred.interval(alpha=0.95)

            preds5[:, (T - 1)] = pred5
            preds50[:, (T - 1)] = pred50
            preds95[:, (T - 1)] = pred95

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

        target = target[x]

        train_size = len(X_train)

        dif = len(x) - len(y5)

        if dif < 0:
            y5 = y5[: len(y5) + dif]
            y50 = y50[: len(y50) + dif]
            y95 = y95[: len(y95) + dif]

        df_pred = pd.DataFrame()
        df_pred["target"] = target
        df_pred["date"] = x
        df_pred["lower"] = y5
        df_pred["median"] = y50
        df_pred["upper"] = y95
        df_pred["train_size"] = [train_size] * len(df_pred)

    else:
        x = pd.period_range(
            start=df_lag.index[1], end=df_lag.index[-1], freq="D"
        ).to_timestamp()

        target = target[x]

        x = np.array(x)

        df_pred = pd.DataFrame()
        df_pred["target"] = target
        df_pred["date"] = x
        df_pred["lower"] = [0.0]
        df_pred["median"] = [0.0]
        df_pred["upper"] = [0.0]
        df_pred["train_size"] = [len(X_train)]

    return df_pred, models, X_train, targets


def train_eval_single_canton(
    target_curve_name,
    canton,
    predictors,
    cantons=None,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    split=0.75,
    horizon=14,
    maxlag=14,
    parameters_model=params_model,
):

    """
    Function to train and evaluate the model for one georegion.

    Important:
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients
    :params target_curve_name: string to indicate the target column of the predictions
    :params canton: canton of interest
    :params predictors: variables that  will be used in model
    :params cantons: list of other cantons whose data will be used as predictors
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed (7 day moving average) or not
    :params ini_date: Determines the beggining of the train dataset
    :params split: float. Determines which percentage of the data will be used to train the model
    :params horizon_forecast: int. Number of days that will be predicted
    :params maxlag: int. Number of the last days that will be used to forecast the next days
    :params parameters_model: dict with the params that will be used in the ngboost
                             regressor model.

    returns: Dataframe.
    """

    target_name = f"{target_curve_name}_{canton}"

    if cantons != None:
        cluster = cantons

    else:
        cluster = [canton]

    df = get_cluster_data(
        "switzerland", predictors, cluster, vaccine=vaccine, smooth=smooth
    )

    df = df.fillna(0)

    df_pred = rolling_predictions(
        target_name,
        df,
        ini_date=ini_date,
        split=split,
        horizon_forecast=horizon,
        maxlag=maxlag,
        kwargs=parameters_model,
    )[0]

    df_pred["canton"] = canton

    return df_pred


def train_eval_all_cantons(
    target_curve_name,
    predictors,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    split=0.75,
    horizon_forecast=14,
    maxlag=14,
    parameters_model=params_model,
):

    """
    Function to make prediction for all the cantons

    Important:
    * By default in the function, for each canton is used your own data as predictors
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    :params target_curve_name: string to indicate the target column of the predictions
    :params predictors: variables that  will be used in model
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed or not
    :params ini_date: Determines the beggining of the train dataset
    :params split: float. Determines which percentage of the data will be used to train the model
    :params horizon_forecast: int. Number of days that will be predicted
    :params maxlag: int. Number of the last days that will be used to forecast the next days
    :params parameters_model: dict with the params that will be used in the ngboost
                             regressor model.

    returns: Dataframe.
    """

    df_all = pd.DataFrame()

    all_cantons = [
        "LU",
        "JU",
        "AR",
        "VD",
        "NE",
        "FR",
        "GL",
        "GR",
        "SG",
        "AI",
        "NW",
        "ZG",
        "SH",
        "GE",
        "BL",
        "BE",
        "BS",
        "TI",
        "UR",
        "AG",
        "TG",
        "SZ",
        "SO",
        "ZH",
        "VS",
        "OW",
    ]

    for canton in all_cantons:

        df = get_cluster_data(
            "switzerland", predictors, [canton], vaccine=vaccine, smooth=smooth
        )

        df = df.fillna(0)

        target_name = f"{target_curve_name}_{canton}"

        df_pred = rolling_predictions(
            target_name,
            df,
            ini_date=ini_date,
            split=split,
            horizon_forecast=horizon_forecast,
            maxlag=maxlag,
            kwargs=parameters_model,
        )[0]

        df_pred["canton"] = canton

        df_all = pd.concat([df_all, df_pred])

    return df_all


def training_model(
    target_name,
    data,
    ini_date="2020-03-01",
    horizon_forecast=14,
    maxlag=14,
    save=True,
    path="../opt/models/saved_models/ml",
    kwargs=params_model,
):

    """
    Function to train multiple ngboost regressor models given a dataset and a target column.
    This function will train multiple models, each one specilist in predict the X + n
    days, of the target column, where n is in the range [1, number of days that you
                                                         want predict].
    This function will train the model with all the data available and will save the model
    that will be used to make forecasts.

    :params target_name:string. Name of the target column.
    :params data: dataframe. Dataframe with features and target column.
    :params ini_date: string. Determines the beggining of the train dataset
    :params horizon_forecast: int. Number of days that will be predicted
    :params max_lag: int. Number of the last days that will be used to forecast the next days
    :params save:boolean. Indicates if the models will be saved in some path or not
    :params path: string. Indicates the path where the models will be saved
    :params kwargs: dict with the params that will be used in the ngboost
                             regressor model.

    returns: list with the {horizon_forecast} models saved
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

    models = []

    # condition to only apply the model for datas with values different of 0
    if np.sum(target) > 0.0:

        # targets
        targets = {}

        for T in np.arange(1, horizon_forecast + 1, 1):
            targets[T] = target.shift(-(T))[:-(T)]

        X_train = df_lag

        for T in range(1, horizon_forecast + 1):

            tgt = targets[T]

            i = 0

            while i < len(tgt):
                if tgt[i] <= 0:

                    tgt[i] = 0.01
                i = i + 1

            model = NGBRegressor(**kwargs)

            model.fit(X_train.iloc[: len(tgt)], tgt)

            if save:
                if path != None:
                    dump(model, f"{path}/ngboost_{target_name}_{T}D.joblib")
                else:
                    dump(model, f"ngboost_{target_name}_{T}D.joblib")

            models.append(model)

    return models


def train_single_canton(
    target_curve_name,
    canton,
    predictors,
    cantons,
    path="../opt/models/saved_models/ml",
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    horizon_forecast=14,
    maxlag=14,
    parameters_model=params_model,
):

    """
    Function to train and evaluate the model for one georegion

    Important:
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    :params canton: canton of interest
    :params predictors: variables that  will be used in model
    :params cantons: list of other cantons whose data will be used as predictors
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed or not
    :params ini_date: Determines the beggining of the train dataset
    :params path: Determines  where the model trained will be saved
    :params horizon_forecast: int. Number of days that will be predicted
    :params maxlag: int. Number of the last days that will be used to forecast the next days
    :params parameters_model: dict with the params that will be used in the ngboost
                             regressor model.
    :returns: None
    """

    target_name = f"{target_curve_name}_{canton}"

    if cantons != None:
        cluster = cantons

    else:
        cluster = [canton]

    df = get_cluster_data(
        "switzerland", predictors, cluster, vaccine=vaccine, smooth=smooth
    )
    df = df.fillna(0)

    training_model(
        target_name,
        df,
        ini_date,
        path=path,
        horizon_forecast=horizon_forecast,
        maxlag=maxlag,
        kwargs=parameters_model,
    )

    return


def train_all_cantons(
    target_curve_name,
    predictors,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    horizon_forecast=14,
    maxlag=14,
    parameters_model=params_model,
    path="saved_models",
):

    """
    Function to train and evaluate the model for alk the cantons in switzerland

    Important:
    * By default in the function for each canton is used your own data as predictors
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    :params target_curve_name: string to indicate the target column of the predictions
    :params predictors: variables that  will be used in model
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed or not
    :params ini_date: Determines the beggining of the train dataset
    :params horizon_forecast: int. Number of days that will be predicted
    :params maxlag: int. Number of the last days that will be used to forecast the next days
    :params parameters_model: dict with the params that will be used in the ngboost
                             regressor model
    :params path: Determines  where the models trained will be saved

    returns: None
    """

    all_cantons = [
        "LU",
        "JU",
        "AR",
        "VD",
        "NE",
        "FR",
        "GL",
        "GR",
        "SG",
        "AI",
        "NW",
        "ZG",
        "SH",
        "GE",
        "BL",
        "BE",
        "BS",
        "TI",
        "UR",
        "AG",
        "TG",
        "SZ",
        "SO",
        "ZH",
        "VS",
        "OW",
    ]

    for canton in all_cantons:

        df = get_cluster_data(
            "switzerland", predictors, [canton], vaccine=vaccine, smooth=smooth
        )

        df = df.fillna(0)

        target_name = f"{target_curve_name}_{canton}"

        training_model(
            target_name,
            df,
            ini_date,
            horizon_forecast=horizon_forecast,
            maxlag=maxlag,
            kwargs=parameters_model,
            path=path,
        )

    return


def rolling_forecast(
    target_name,
    data,
    horizon_forecast=14,
    maxlag=14,
    path="../opt/models/saved_models/ml",
):

    """
    Function to load multiple ngboost regressors model trained with the function
    `training_model` and make the forecast.

    Important:
    horizon_forecast and max_lag need have the same value used in training_model
    Only the last row of the dataset will be used to forecast the next
    horizon_forecast days.

    :params target_name:string. Name of the target column.
    :params data: dataframe. Dataframe with features and target column.
    :params horizon_forecast: int. Number of days that will be predicted
    :params maxlag: int. Number of the last days that will be used to forecast the next days
    :params path: string. Indicates where the model is save to the function load the model

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

    forecasts5 = []
    forecasts50 = []
    forecasts95 = []

    X_train = df_lag.iloc[:-1]

    for T in range(1, horizon_forecast + 1):

        if path != None:
            model = load(f"{path}/ngboost_{target_name}_{T}D.joblib")

        else:
            model = load(f"ngboost_{target_name}_{T}D.joblib")

        forecast = model.pred_dist(df_lag.iloc[-1:])

        forecast50 = forecast.median()

        forecast5, forecast95 = forecast.interval(alpha=0.95)

        forecasts5.append(forecast5)
        forecasts50.append(forecast50)
        forecasts95.append(forecast95)

    forecast_dates = []

    last_day = datetime.strftime((df_lag.index)[-1], "%Y-%m-%d")

    a = datetime.strptime(last_day, "%Y-%m-%d")

    for i in np.arange(1, horizon_forecast + 1):

        d_i = datetime.strftime(a + timedelta(days=int(i)), "%Y-%m-%d")

        forecast_dates.append(datetime.strptime(d_i, "%Y-%m-%d"))

    df_for = pd.DataFrame()
    df_for["date"] = forecast_dates
    df_for["lower"] = np.array(forecasts5)
    df_for["median"] = np.array(forecasts50)
    df_for["upper"] = np.array(forecasts95)

    return df_for


def forecast_single_canton(
    target_curve_name,
    canton,
    predictors,
    cantons=None,
    vaccine=True,
    smooth=True,
    horizon_forecast=14,
    maxlag=14,
    path="../opt/models/saved_models/ml",
):
    """
    Function to make the forecast for one canton

    Important:
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    :params target_curve_name: string to indicate the target column of the predictions
    :params canton: string to indicate the interest canton
    :params predictors: variables that  will be used in model
    :params cantons: list of other cantons whose data will be used as predictors
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed or not
    :params horizon_forecast: int. Number of days that will be predicted
    :params maxlag: int. Number of the last days that will be used to forecast the next days
    :params path: string. Indicates where the models trained are saved

    returns: Dataframe with the forecast for the canton
    """

    if cantons != None:
        cluster = cantons

    else:
        cluster = [canton]

    df = get_cluster_data(
        "switzerland", predictors, cluster, vaccine=vaccine, smooth=smooth
    )

    df = df.fillna(0)

    target_name = f"{target_curve_name}_{canton}"

    df_for = rolling_forecast(
        target_name,
        df,
        horizon_forecast=horizon_forecast,
        maxlag=maxlag,
        path=path,
    )
    df_for["canton"] = canton

    return df_for


def forecast_all_cantons(
    target_curve_name,
    predictors,
    vaccine=True,
    smooth=True,
    horizon_forecast=14,
    maxlag=14,
    path="../opt/models/saved_models/ml",
):
    """
    Function to make the forecast for all the cantons

    Important:
    * By default in the function for each canton is used your own data as predictors
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    :params target_curve_name: string to indicate the target column of the predictions
    :params predictors: variables that  will be used in model
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed or not
    :params horizon_forecast: int. Number of days that will be predicted
    :params maxlag: int. Number of the last days that will be used to forecast the next days
    :params path: string. Indicates where the models trained are saved.

    returns: Dataframe with the forecast for all the cantons
    """
    df_all = pd.DataFrame()

    all_cantons = [
        "LU",
        "JU",
        "AR",
        "VD",
        "NE",
        "FR",
        "GL",
        "GR",
        "SG",
        "AI",
        "NW",
        "ZG",
        "SH",
        "GE",
        "BL",
        "BE",
        "BS",
        "TI",
        "UR",
        "AG",
        "TG",
        "SZ",
        "SO",
        "ZH",
        "VS",
        "OW",
    ]

    for canton in all_cantons:

        df = get_cluster_data(
            "switzerland", predictors, [canton], vaccine=vaccine, smooth=smooth
        )

        df = df.fillna(0)

        target_name = f"{target_curve_name}_{canton}"

        df_for = rolling_forecast(
            target_name,
            df,
            horizon_forecast=horizon_forecast,
            maxlag=maxlag,
            path=path,
        )
        df_for["canton"] = canton

        df_all = pd.concat([df_all, df_for])

    return df_all
