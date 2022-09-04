#!/usr/bin/env python3
"""
The functions in this module allow the application of the
ngboost regressor model for time series. There are separate functions to train and evaluate (separate
the data in train and test datasets), train with all the data available, and make
forecasts. Also, there are functions to apply these methods in just one canton or all
the cantons of switzerland.
"""

import pandas as pd
from ngboost.distns import LogNormal
from ngboost.learners import default_tree_learner
from ngboost.scores import LogScore

from epigraphhub.analysis.clustering import compute_clusters
from epigraphhub.data.foph import get_cluster_data, get_georegion_data
from epigraphhub.models.ngboost_models import forecast_ngb, train_eval_ngb, train_ngb

all_cantons = [
    "AG",
    "AI",
    "AR",
    "BE",
    "BL",
    "BS",
    "GL",
    "GR",
    "LU",
    "NW",
    "OW",
    "SG",
    "SH",
    "SO",
    "SZ",
    "TG",
    "UR",
    "ZG",
    "ZH",
    "FR",
    "JU",
    "NE",
    "TI",
    "VD",
    "VS",
    "GE",
]


params_model = {
    "Base": default_tree_learner,
    "Dist": LogNormal,
    "Score": LogScore,
    "natural_gradient": True,
    "verbose": False,
    "col_sample": 0.9,
    "n_estimators": 100,
    "learning_rate": 0.05,
    "validation_fraction": 0.15,
    "early_stopping_rounds": 10,
}


def get_clusters_swiss(t=0.3):
    """
    Params to get the list of clusters computed by the compute_cluster function.
    :params t: float. Thereshold used in the clusterization.
    :returns: Array with the clusters computed.
    """
    df = get_georegion_data(
        "switzerland",
        "foph_cases_d",
        "All",
        ["datum", '"georegion"', "entries"],
        "georegion",
    )

    df.set_index("datum", inplace=True)
    df.index = pd.to_datetime(df.index)

    clusters = compute_clusters(
        df,
        ["georegion", "entries"],
        t=t,
        drop_georegions=["CH", "FL", "CHFL"],
        plot=False,
    )[1]

    return clusters


def get_cluster_by_canton(canton):
    """
    Function to return the cluster that contains a specific canton.
    :params canton: string. Name (two letters code) of the canton.
    :returns: list with the cluster that contains the canton.
    """

    clusters = get_clusters_swiss()

    cluster_canton = list(filter(lambda x: canton in x, clusters))

    return list(cluster_canton[0])


def remove_zeros(tgt):
    """
    Function to remove the zeros of the target curve. It needs to be done to us be able
    to use the LogNormal dist.
    :params tgt: array.
    """

    tgt[tgt == 0] = 0.01

    return tgt


def train_eval_single_canton(
    target_curve_name,
    canton,
    predictors,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    end_train_date=None,
    end_date=None,
    ratio=0.75,
    parameters_model=params_model,
    predict_n=14,
    look_back=14,
):

    """
    Function to train and evaluate the model for one georegion.

    Important:
    * By default the function is using the clustering cantons and the data since 2020
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    :params canton: canton of interest
    :params predictors: variables that  will be used in model
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed (7 day moving average) or not
    :params ini_date: string. Determines the beggining of the train dataset
    :params end_train_date: string. Determines the beggining of end of train dataset. If end_train_date
                           is not None, then ratio isn't used.
    :params end_date: string. Determines the end of the dataset used in validation.
    :params ratio: float. Determines which percentage of the data will be used to train the model
    :params parameters_model: dict with the params that will be used in the ngboost
                             regressor model.
    :params predict_n: int. Number of days that will be predicted.
    :params look_back: int. Number of the last days that will be used to forecast the next days.

    returns: Dataframe.
    """

    cluster_canton = [canton]  # get_cluster_by_canton(canton)

    target_name = f"{target_curve_name}_{canton}"

    df = get_cluster_data(
        "switzerland", predictors, cluster_canton, vaccine=vaccine, smooth=smooth
    )

    df = df.fillna(0)

    df[target_name] = remove_zeros(df[target_name].values)

    if any(df[target_name] > 1):

        df_pred = train_eval_ngb(
            target_name,
            df,
            ini_date=ini_date,
            end_train_date=end_train_date,
            end_date=end_date,
            ratio=ratio,
            predict_n=predict_n,
            look_back=look_back,
            kwargs=parameters_model,
        )

        df_pred["canton"] = [target_name[-2:]] * len(df_pred)

    else:
        df_pred = pd.DataFrame()
        df_pred["target"] = df[target_name]
        df_pred["date"] = 0
        df_pred["lower"] = 0
        df_pred["median"] = 0
        df_pred["upper"] = 0
        df_pred["canton"] = target_name[-2:]

    return df_pred


def train_eval_all_cantons(
    target_curve_name,
    predictors,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    end_train_date=None,
    end_date=None,
    ratio=0.75,
    parameters_model=params_model,
    predict_n=14,
    look_back=14,
    all_cantons=all_cantons,
):

    """
    Function to make prediction for all the cantons

    Important:
    * By default the function is using the clustering cantons and the data since 2020
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    :params target_curve_name: string to indicate the target column of the predictions
    :params predictors: variables that  will be used in model
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed or not
    :params ini_date: string. Determines the beggining of the train dataset
    :params end_train_date: string. Determines the beggining of end of train dataset. If end_train_date
                           is not None, then ratio isn't used.
    :params end_date: string. Determines the end of the dataset used in validation.
    :params ratio: float. Determines which percentage of the data will be used to train the model
    :params parameters_model: dict with the params that will be used in the ngboost
                             regressor model.
    :params predict_n: int. Number of days that will be predicted.
    :params look_back: int. Number of the last days that will be used to forecast the next days.

    returns: Dataframe.
    """

    df_all = pd.DataFrame()

    clusters = get_clusters_swiss()

    for cluster in clusters:

        df = get_cluster_data(
            "switzerland", predictors, list(cluster), vaccine=vaccine, smooth=smooth
        )

        df = df.fillna(0)

        for canton in cluster:

            target_name = f"{target_curve_name}_{canton}"

            df_c = df.copy()

            df_c[target_name] = remove_zeros(df_c[target_name].values)

            if any(df_c[target_name] > 1):

                df_pred = train_eval_ngb(
                    target_name,
                    df_c,
                    ini_date=ini_date,
                    end_train_date=end_train_date,
                    end_date=end_date,
                    ratio=ratio,
                    predict_n=predict_n,
                    look_back=look_back,
                    kwargs=parameters_model,
                )

                df_pred["canton"] = canton

            else:
                df_pred = pd.DataFrame()
                df_pred["target"] = df[target_name]
                df_pred["date"] = 0
                df_pred["lower"] = 0
                df_pred["median"] = 0
                df_pred["upper"] = 0
                df_pred["canton"] = canton

            df_all = pd.concat([df_all, df_pred])

    return df_all


def train_single_canton(
    target_curve_name,
    canton,
    predictors,
    path="../opt/models/saved_models/ml",
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    end_date=None,
    parameters_model=params_model,
    predict_n=14,
    look_back=14,
):

    """
    Function to train and evaluate the model for one georegion

    Important:
    * By default the function is using the clustering cantons and the data since 2020
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    :params canton: canton of interest
    :params predictors: variables that  will be used in model
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed or not
    :params ini_date: Determines the beggining of the train dataset
    :params path: Determines  where the model trained will be saved
    :params update_data: Determines if the data from the Geneva hospital will be used.
                        this params only is used when canton = GE and target_curve_name = hosp.
    :params parameters_model: dict with the params that will be used in the ngboost
                             regressor model.
    :params predict_n: int. Number of days that will be predicted.
    :params look_back: int. Number of the last days that will be used to forecast the next days.

    :returns: None
    """

    cluster_canton = [canton]  # get_cluster_by_canton(canton)

    target_name = f"{target_curve_name}_{canton}"

    # getting the data
    df = get_cluster_data(
        "switzerland", predictors, list(cluster_canton), vaccine=vaccine, smooth=smooth
    )
    # filling the nan values with 0
    df = df.fillna(0)

    df[target_name] = remove_zeros(df[target_name].values)

    if any(df[target_name] > 1):

        train_ngb(
            target_name,
            df,
            ini_date=ini_date,
            path=path,
            end_date=end_date,
            predict_n=predict_n,
            look_back=look_back,
            kwargs=parameters_model,
        )

    else:
        print(
            f"The model to forecast {target_name} was not trained, since the series has no value bigger than one."
        )

    return


def train_all_cantons(
    target_curve_name,
    predictors,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    parameters_model=params_model,
    predict_n=14,
    look_back=14,
    path=None,
):

    """
    Function to train and evaluate the model for alk the cantons in switzerland

    Important:
    * By default the function is using the clustering cantons and the data since 2020
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    :params target_curve_name: string to indicate the target column of the predictions
    :params predictors: variables that  will be used in model
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed or not
    :params ini_date: Determines the beggining of the train dataset
    :params parameters_model: dict with the params that will be used in the ngboost
                             regressor model.
    :params predict_n: int. Number of days that will be predicted.
    :params look_back: int. Number of the last days that will be used to forecast the next days.

    returns: Dataframe with the forecast for all the cantons
    """

    clusters = get_clusters_swiss()

    for cluster in clusters:

        df = get_cluster_data(
            "switzerland", predictors, list(cluster), vaccine=vaccine, smooth=smooth
        )

        df = df.fillna(0)

        for canton in cluster:

            df_c = df.copy()

            target_name = f"{target_curve_name}_{canton}"

            df_c[target_name] = remove_zeros(df_c[target_name].values)

            if any(df_c[target_name] > 1):

                train_ngb(
                    target_name,
                    df_c,
                    ini_date,
                    predict_n=predict_n,
                    look_back=look_back,
                    kwargs=parameters_model,
                    path=path,
                )

    return


def forecast_single_canton(
    target_curve_name,
    canton,
    predictors,
    vaccine=True,
    smooth=True,
    end_date=None,
    path="../opt/models/saved_models/ml",
    predict_n=14,
    look_back=14,
):
    """
    Function to make the forecast for one canton

    Important:
    * By default the function is using the clustering cantons and the data since 2020
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    :params target_curve_name: string to indicate the target column of the predictions
    :params canton: string to indicate the interest canton
    :params predictors: variables that  will be used in model
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed or not
    :params end_date: string. Determines from what day the forecast will be computed.
    :params path: string. Indicates where the models trained are saved.
    :params predict_n: int. Number of days that will be predicted.
    :params look_back: int. Number of the last days that will be used to forecast the next days.

    returns: Dataframe with the forecast for all the cantons
    """

    cluster_canton = [canton]  # get_cluster_by_canton(canton)

    df = get_cluster_data(
        "switzerland", predictors, list(cluster_canton), vaccine=vaccine, smooth=smooth
    )

    df = df.fillna(0)

    target_name = f"{target_curve_name}_{canton}"

    df_for = forecast_ngb(
        target_name,
        df,
        end_date=end_date,
        predict_n=predict_n,
        look_back=look_back,
        path=path,
    )

    return df_for


def forecast_all_cantons(
    target_curve_name,
    predictors,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    path="../opt/models/saved_models/ml",
    predict_n=14,
    look_back=14,
):
    """
    Function to make the forecast for all the cantons

    Important:
    * By default the function is using the clustering cantons and the data since 2020
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    :params target_curve_name: string to indicate the target column of the predictions
    :params predictors: variables that  will be used in model
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed or not
    :params end_date: string. Determines from what day the forecast will be computed.
    :params path: string. Indicates where the models trained are saved.

    returns: Dataframe with the forecast for all the cantons
    """
    df_all = pd.DataFrame()

    clusters = get_clusters_swiss()

    for cluster in clusters:

        df = get_cluster_data(
            "switzerland", predictors, list(cluster), vaccine=vaccine, smooth=smooth
        )

        df = df.fillna(0)

        for canton in cluster:

            target_name = f"{target_curve_name}_{canton}"

            df_for = forecast_ngb(
                target_name,
                df,
                predict_n=predict_n,
                look_back=look_back,
                path=path,
            )

            df_all = pd.concat([df_all, df_for])

    return df_all
