#!/usr/bin/env python3
"""
The functions in this module allow the application of the ngboost
regressor model for time series. There are separate functions to train
and evaluate (separate the data in train and test datasets), train with
all the data available, and make forecasts. Also, there are functions to
apply these methods in just one canton or all the cantons of
switzerland.

"""
import pandas as pd
from ngboost.distns import LogNormal
from ngboost.learners import default_tree_learner
from ngboost.scores import LogScore

from epigraphhub.analysis.clustering import compute_clusters
from epigraphhub.analysis.forecast_models.ngboost_models import NGBModel
from epigraphhub.data.foph.viz import get_cluster_data, get_data_by_location

params_model = {
    "Base": default_tree_learner,
    "Dist": LogNormal,
    "Score": LogScore,
    "natural_gradient": True,
    "verbose": False,
    "col_sample": 0.9,
    "n_estimators": 100,
    "learning_rate": 0.05,
}


def get_clusters_swiss(t=0.3, end_date=None):
    """
    Params to get the list of clusters computed by the compute_cluster
    function.

    Parameters
    ----------
    t : float
        Thereshold used in the clusterization.
    end_date : str
        Indicates the last day used to compute the cluster.

    Returns
    -------
    Array
        Array with the clusters computed.
    """

    df = get_data_by_location(
        "switzerland",
        "foph_cases_d",
        "All",
        ["datum", '"georegion"', "entries"],
        "georegion",
    )

    df.set_index("datum", inplace=True)
    df.index = pd.to_datetime(df.index)

    if end_date != None:
        df = df.loc[:end_date]

    clusters = compute_clusters(
        df,
        ["georegion", "entries"],
        t=t,
        drop_values=["CH", "FL", "CHFL"],
        plot=False,
        smooth=True,
    )[1]

    return clusters


def get_cluster_by_canton(canton):
    """
    Function to return the cluster that contains a specific canton.

    Parameters
    ----------
    canton : str
        Name (two letters code) of the canton.

    Returns
    -------
    List
        List with the cluster that contains the canton.
    """

    clusters = get_clusters_swiss(t=0.6)

    cluster_canton = list(filter(lambda x: canton in x, clusters))

    return list(cluster_canton[0])


def remove_zeros(tgt):
    """
    Function to remove the zeros of the target curve. It needs to be
    done to us be able to use the LogNormal dist.

    Parameters
    ----------
    tgt : array

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
    ratio_val=0.15,
    early_stop=5,
    parameters_model=params_model,
    predict_n=14,
    look_back=14,
):
    """
    Function to train and evaluate the model for one georegion.

    Important: * By default the function is using the clustering cantons
    and the
      data since 2020.
    * For the predictor hospCapacity is used as predictor the column
      ICU_Covid19Patients.

    Parameters
    ----------
    target_curve_name : str canton : str
        Canton of interest.
    predictors : list
        Variables that will be used in model.
    vaccine : bool
        It determines if the vaccine data from owid will be used or not.
    smooth : bool
        It determines if data will be smoothed (7 day moving average) or
        not.
    ini_date : str
        Determines the beggining of the train dataset.
    end_train_date : str
        Determines the beggining of end of train dataset. If is not
        None, then ratio isn't used.
    end_date : str
        Determines the end of the dataset used in validation.
    ratio : float
        Determines which percentage of the data will be used to train
        the model.
    ratio_val : float
        Determines which percentage of the train data will be used as
        validation data.
    early_stop : int
        This parameter will finish the model's training after
        {early_stop} iterations without improving the model in the
        validation data.
    parameters_model : dict
        Dict with the params that will be used in the ngboost regressor
        model.
    predict_n : int
        Number of days that will be predicted.
    look_back : int
        Number of the last days that will be used to forecast the next
        days.

    Returns
    -------
    pd.DataFrame
        The return is a pandas DataFrame.

    """

    cluster_canton = [canton]  # get_cluster_by_canton(canton)

    target_name = f"{target_curve_name}_{canton}"

    df = get_cluster_data(
        "switzerland", predictors, cluster_canton, vaccine=vaccine, smooth=smooth
    )

    df = df.fillna(0)

    df[target_name] = remove_zeros(df[target_name].values)

    ngb_m = NGBModel(
        look_back=look_back,
        predict_n=predict_n,
        validation_split=ratio_val,
        early_stop=early_stop,
        params_model=parameters_model,
    )

    if any(df[target_name] > 1):

        df_pred = ngb_m.train_eval(
            target_name,
            df,
            ini_date=ini_date,
            end_train_date=end_train_date,
            end_date=end_date,
            ratio=ratio,
            save=False,
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
):

    """
    Function to make prediction for all the cantons.

    Important:
    * By default the function is using the clustering cantons and the
      data since 2020.
    * For the predictor hospCapacity is used as predictor the column
      ICU_Covid19Patients.

    Parameters
    ----------
    target_curve_name : str
        Indicates the target column of the predictions.
    predictors : str
        Variables that will be used in model.
    vaccine : bool
        It determines if the vaccine data from owid will be used or not.
    smooth : bool
        It determines if data will be smoothed or not.
    ini_date : str
        Determines the beggining of the train dataset.
    end_train_date : str
        Determines the beggining of end of train dataset. If is not
        None, then ratio isn't used.
    end_date : str
        Determines the end of the dataset used in validation.
    ratio : float
        Determines which percentage of the data will be used to train
        the model.
    parameters_model : dict
        Dict with the params that will be used in the ngboost regressor
        model.
    predict_n : int
        Number of days that will be predicted.
    look_back : int
        Number of the last days that will be used to forecast the next
        days.

    Returns
    -------
    pd.DataFrame
        The return is a pandas DataFrame.

    """

    df_all = pd.DataFrame()

    clusters = get_clusters_swiss(t=0.6)

    for cluster in clusters:

        df = get_cluster_data(
            "switzerland", predictors, list(cluster), vaccine=vaccine, smooth=smooth
        )

        df = df.fillna(0)

        for canton in cluster:

            target_name = f"{target_curve_name}_{canton}"

            df_c = df.copy()

            df_c[target_name] = remove_zeros(df_c[target_name].values)

            ngb_m = NGBModel(
                look_back=look_back,
                predict_n=predict_n,
                validation_split=0.15,
                early_stop=10,
                params_model=parameters_model,
            )

            if any(df_c[target_name] > 1):

                df_pred = ngb_m.train_eval(
                    target_name,
                    df_c,
                    ini_date=ini_date,
                    end_train_date=end_train_date,
                    end_date=end_date,
                    ratio=ratio,
                    save=False,
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
    Function to train and evaluate the model for one georegion.

    Important: * By default the function is using the clustering cantons
    and the
      data since 2020.
    * For the predictor hospCapacity is used as predictor the column
      ICU_Covid19Patients.

    Parameters
    ----------
    canton : str
        Canton of interest.
    predictors : str
        Variables that will be used in model.
    vaccine : bool
        It determines if the vaccine data from owid will be used or not.
    smooth : bool
        It determines if data will be smoothed or not.
    ini_date : str
        Determines the beggining of the train dataset
    path : str
        Determines where the model trained will be saved.
    update_data : bool
        Determines if the data from the Geneva hospital will be used.
        This params only is used when canton = GE and target_curve_name
        = hosp.
    parameters_model : dict
        Dict with the params that will be used in the ngboost regressor
        model.
    predict_n : int
        Number of days that will be predicted.
    look_back : int
        Number of the last days that will be used to forecast the next
        days.

    Returns
    -------
    None
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

    ngb_m = NGBModel(
        look_back=look_back,
        predict_n=predict_n,
        validation_split=0.15,
        early_stop=10,
        params_model=parameters_model,
    )

    if any(df[target_name] > 1):

        ngb_m.train(
            target_name,
            df,
            ini_date=ini_date,
            path=path,
            end_date=end_date,
            save=True,
            name=f"ngboost_{target_curve_name}_{canton}",
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
    end_date=None,
    parameters_model=params_model,
    predict_n=14,
    look_back=14,
    path=None,
):

    """
    Function to train and evaluate the model for all the cantons in
    switzerland.

    Important:
    * By default the function is using the clustering cantons and the
      data since 2020.
    * For the predictor hospCapacity is used as predictor the column
      ICU_Covid19Patients.

    Parameters
    ----------
    target_curve_name : str
        Indicates the target column of the predictions.
    predictors : str
        Variables that will be used in model.
    vaccine : bool
        It determines if the vaccine data from owid will be used or not.
    smooth : bool
        It determines if data will be smoothed or not.
    ini_date : str
        Determines the beggining of the train dataset.
    parameters_model : dict
        Dict with the params that will be used in the ngboost regressor
        model.
    predict_n : int
        Number of days that will be predicted.
    look_back : int
        Number of the last days that will be used to forecast the next
        days.

    Returns
    -------
    pd.DataFrame
        Dataframe with the forecast for all the cantons.
    """

    clusters = get_clusters_swiss(t=0.6)

    for cluster in clusters:

        df = get_cluster_data(
            "switzerland", predictors, list(cluster), vaccine=vaccine, smooth=smooth
        )

        df = df.fillna(0)

        for canton in cluster:

            df_c = df.copy()

            target_name = f"{target_curve_name}_{canton}"

            df_c[target_name] = remove_zeros(df_c[target_name].values)

            ngb_m = NGBModel(
                look_back=look_back,
                predict_n=predict_n,
                validation_split=0.15,
                early_stop=10,
                params_model=parameters_model,
            )

            if any(df_c[target_name] > 1):
                ngb_m.train(
                    target_name,
                    df_c,
                    ini_date=ini_date,
                    path=path,
                    end_date=end_date,
                    save=True,
                    name=f"ngboost_{target_name}",
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
    Function to make the forecast for one canton.

    Important:
    * By default the function is using the clustering cantons and the
      data since 2020.
    * For the predictor hospCapacity is used as predictor the column
      ICU_Covid19Patients.

    Parameters
    ----------
    target_curve_name : str
        Indicates the target column of the predictions.
    canton : str
        Indicates the interest canton.
    predictors : str
        Variables that will be used in model.
    vaccine : bool
        It determines if the vaccine data from owid will be used or not.
    smooth : bool
        It determines if data will be smoothed or not.
    end_date : str
        Determines from what day the forecast will be computed.
    path : str
        Indicates where the models trained are saved.
    predict_n : int
        Number of days that will be predicted.
    look_back : int
        Number of the last days that will be used to forecast the next
        days.

    Returns
    -------
    pd.DataFrame
        Dataframe with the forecast for one canton.
    """

    cluster_canton = [canton]  # get_cluster_by_canton(canton)

    df = get_cluster_data(
        "switzerland", predictors, list(cluster_canton), vaccine=vaccine, smooth=smooth
    )

    df = df.fillna(0)

    target_name = f"{target_curve_name}_{canton}"

    ngb_m = NGBModel(
        look_back=look_back, predict_n=predict_n, validation_split=0.15, early_stop=10
    )

    df_for = ngb_m.forecast(
        df, end_date=end_date, path=path, name=f"ngboost_{target_name}"
    )

    return df_for


def forecast_all_cantons(
    target_curve_name,
    predictors,
    vaccine=True,
    smooth=True,
    path="../opt/models/saved_models/ml",
    predict_n=14,
    look_back=14,
):
    """
    Function to make the forecast for all the cantons.

    Important:
    * By default the function is using the clustering cantons and the
      data since 2020.
    * For the predictor hospCapacity is used as predictor the column
      ICU_Covid19Patients.

    Parameters
    ----------
    target_curve_name : str
        Indicates the target column of the predictions.
    predictors : str
        Variables that  will be used in model.
    vaccine : bool
        It determines if the vaccine data from owid will be used or not.
    smooth : bool
        It determines if data will be smoothed or not.
    end_date : str
        Determines from what day the forecast will be computed.
    path : str
        Indicates where the models trained are saved.

    Returns
    -------
    pd.DataFrame
        Dataframe with the forecast for all the cantons.
    """

    df_all = pd.DataFrame()

    clusters = get_clusters_swiss(t=0.6)

    for cluster in clusters:

        df = get_cluster_data(
            "switzerland", predictors, list(cluster), vaccine=vaccine, smooth=smooth
        )

        df = df.fillna(0)

        for canton in cluster:

            target_name = f"{target_curve_name}_{canton}"

            ngb_m = NGBModel(
                look_back=look_back,
                predict_n=predict_n,
                validation_split=0.15,
                early_stop=10,
            )

            df[target_name] = remove_zeros(df[target_name].values)

            if any(df[target_name] > 1):
                df_for = ngb_m.forecast(df, path=path, name=f"ngboost_{target_name}")

                df_for["canton"] = canton

                df_all = pd.concat([df_all, df_for])

    return df_all


def save_to_database(df, table_name, engine):
    df.to_sql(table_name, engine, schema="switzerland", if_exists="replace")
