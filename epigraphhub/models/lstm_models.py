#!/usr/bin/env python3
"""
Created on Mon Jan 31 16:20:05 2022

@author: eduardoaraujo
"""

import pickle
from datetime import datetime, timedelta
from time import time

import numpy as np
import pandas as pd
import tensorflow.keras as keras
from matplotlib import pyplot as plt

# import math
# import os
# import shap
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
)
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.utils import plot_model

from epigraphhub.analysis.clustering import compute_clusters
from epigraphhub.data.get_data import get_cluster_data, get_updated_data
from epigraphhub.data.preprocessing import lstm_split_data as split_data
from epigraphhub.data.preprocessing import normalize_data


def build_model(hidden, features, predict_n, look_back=10, batch_size=1):
    """
    Builds and returns the LSTM model with the parameters given
    :param hidden: number of hidden nodes
    :param features: number of variables in the example table
    :param look_back: Number of time-steps to look back before predicting
    :param batch_size: batch size for batch training
    :return:
    """

    inp = keras.Input(
        shape=(look_back, features),
        # batch_shape=(batch_size, look_back, features)
    )
    x = Bidirectional(
        LSTM(
            hidden,
            input_shape=(look_back, features),
            stateful=False,
            kernel_initializer="he_uniform",
            batch_input_shape=(batch_size, look_back, features),
            return_sequences=True,
            activation="relu",
            dropout=0.1,
            recurrent_dropout=0.1,
            implementation=2,
            unit_forget_bias=True,
        ),
        merge_mode="ave",
    )(inp, training=True)
    x = Dropout(0.2)(x, training=True)
    x = Bidirectional(
        LSTM(
            hidden,
            input_shape=(look_back, features),
            stateful=False,
            kernel_initializer="he_uniform",
            batch_input_shape=(batch_size, look_back, features),
            return_sequences=True,
            activation="relu",
            dropout=0.1,
            recurrent_dropout=0.1,
            implementation=2,
            unit_forget_bias=True,
        ),
        merge_mode="ave",
    )(x, training=True)
    x = Dropout(0.2)(x, training=True)
    x = Bidirectional(
        LSTM(
            hidden,
            input_shape=(look_back, features),
            kernel_initializer="he_uniform",
            stateful=False,
            batch_input_shape=(batch_size, look_back, features),
            # return_sequences=True,
            activation="relu",
            dropout=0.1,
            recurrent_dropout=0.1,
            implementation=2,
            unit_forget_bias=True,
        ),
        merge_mode="ave",
    )(x, training=True)
    x = Dropout(0.2)(x, training=True)
    out = Dense(
        predict_n,
        activation="relu",
        kernel_initializer="he_uniform",
        bias_initializer="zeros",
    )(x)
    model = keras.Model(inp, out)

    start = time()
    model.compile(loss="msle", optimizer="adam", metrics=["accuracy", "mape", "mse"])
    print("Compilation Time : ", time() - start)
    plot_model(model, to_file="LSTM_model.png")
    print(model.summary())
    return model


def train(
    model, X_train, Y_train, batch_size=1, epochs=10, path=None, label="GE", save=False
):

    TB_callback = TensorBoard(
        log_dir="./tensorboard",
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        # embeddings_freq=10
    )

    hist = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.15,
        shuffle=False,
        verbose=1,
        callbacks=[TB_callback, EarlyStopping(patience=15)],
    )

    if save:
        if path is None:
            with open("f'history_{label}.pkl", "wb") as f:
                pickle.dump(hist.history, f)
        else:
            with open("f'{path}/history_{label}.pkl", "wb") as f:
                pickle.dump(hist.history, f)

    return hist


def plot_training_history(hist):
    """
    Plot the Loss series from training the model
    :param hist: Training history object returned by "model.fit()"
    """
    df_vloss = pd.DataFrame(hist.history["val_loss"], columns=["val_loss"])
    df_loss = pd.DataFrame(hist.history["loss"], columns=["loss"])
    df_mape = pd.DataFrame(
        hist.history["mean_absolute_percentage_error"], columns=["mape"]
    )
    ax = df_vloss.plot(logy=True)
    df_loss.plot(ax=ax, grid=True, logy=True)
    # df_mape.plot(ax=ax, grid=True, logy=True);
    # P.savefig("{}/LSTM_training_history.png".format(FIG_PATH))


def plot_predicted_vs_data(
    predicted,
    Ydata,
    indice,
    canton,
    pred_window,
    factor,
    split_point=None,
    uncertainty=False,
):
    """
    Plot the model's predictions against data
    :param predicted: model predictions
    :param Ydata: observed data
    :param indice:
    :param label: Name of the locality of the predictions
    :param pred_window:
    :param factor: Normalizing factor for the target variable
    """

    plt.clf()
    if len(predicted.shape) == 2:
        df_predicted = pd.DataFrame(predicted)
    else:
        df_predicted = pd.DataFrame(np.percentile(predicted, 50, axis=2))
        df_predicted25 = pd.DataFrame(np.percentile(predicted, 2.5, axis=2))
        df_predicted975 = pd.DataFrame(np.percentile(predicted, 97.5, axis=2))
        uncertainty = True
    ymax = max(predicted.max() * factor, Ydata.max() * factor)
    plt.vlines(indice[split_point], 0, ymax, "g", "dashdot", lw=2)
    plt.text(indice[split_point + 2], 0.6 * ymax, "Out of sample Predictions")
    # plot only the last (furthest) prediction point
    plt.plot(
        indice[len(indice) - Ydata.shape[0] :],
        Ydata[:, -1] * factor,
        "k-",
        alpha=0.7,
        label="data",
    )
    plt.plot(
        indice[len(indice) - Ydata.shape[0] :],
        df_predicted.iloc[:, -1] * factor,
        "r-",
        alpha=0.5,
        label="median",
    )
    if uncertainty:
        plt.fill_between(
            indice[len(indice) - Ydata.shape[0] :],
            df_predicted25[df_predicted25.columns[-1]] * factor,
            df_predicted975[df_predicted975.columns[-1]] * factor,
            color="b",
            alpha=0.3,
        )

    # plot all predicted points
    # plt.plot(indice[pred_window:], pd.DataFrame(Ydata)[7] * factor, 'k-')
    # for n in range(df_predicted.shape[1] - pred_window):
    #     plt.plot(
    #         indice[n: n + pred_window],
    #         pd.DataFrame(Ydata.T)[n] * factor,
    #         "k-",
    #         alpha=0.7,
    #     )
    #     plt.plot(indice[n: n + pred_window], df_predicted[n] * factor, "r-")
    #     try:
    #         plt.vlines(
    #             indice[n + pred_window],
    #             0,
    #             df_predicted[n].values[-1] * factor,
    #             "b",
    #             alpha=0.2,
    #         )
    #     except IndexError as e:
    #         print(indice.shape, n, df_predicted.shape)
    tag = "_unc" if uncertainty else ""
    plt.grid()
    plt.title(f"Predictions for {canton}")
    plt.xlabel("time")
    plt.ylabel("incidence")
    plt.xticks(rotation=70)
    plt.legend(["data", "predicted"])
    # plt.savefig("lstm_{}{}.png".format(canton, tag), bbox_inches="tight", dpi=300,)
    plt.show()


def predict(model, Xdata, Ydata, uncertainty=False):
    if uncertainty:
        predicted = np.stack([model.predict(Xdata, batch_size=1, verbose=1) for i in range(100)], axis=2)  # type: ignore
    else:
        predicted = model.predict(Xdata, batch_size=1, verbose=1)

    return predicted


def calculate_metrics(pred, ytrue, factor):
    metrics = pd.DataFrame(
        index=(
            "mean_absolute_error",
            "explained_variance_score",
            "mean_squared_error",
            "mean_squared_log_error",
            "median_absolute_error",
            "r2_score",
        )
    )
    for col in range(pred.shape[1]):
        y = ytrue[:, col] * factor
        p = pred[:, col] * factor
        l = [
            mean_absolute_error(y, p),
            explained_variance_score(y, p),
            mean_squared_error(y, p),
            mean_squared_log_error(y, p),
            median_absolute_error(y, p),
            r2_score(y, p),
        ]
        metrics[col] = l
    return metrics


def get_data_model(
    target_curve_name,
    canton,
    cluster,
    predictors,
    look_back=21,
    predict_n=14,
    split=0.75,
    vaccine=True,
    smooth=True,
    updated_data=False,
):
    df = get_cluster_data(predictors, list(cluster), vaccine=vaccine, smooth=smooth)
    # end = time.time()
    # print(end - start)
    # filling the nan values with 0
    df = df.fillna(0)

    # removing the last three days of data to avoid delay in the reporting.

    if f"{target_curve_name}_{canton}" == "hosp_GE":
        if updated_data:
            # atualizando a coluna das Hospitalizações com os dados mais atualizados
            df_new = get_updated_data(smooth)

            df.loc[df_new.index[0] : df_new.index[-1], "hosp_GE"] = df_new.hosp_GE

            # utilizando como último data a data dos dados atualizados:
            df = df.loc[: df_new.index[-1]]

    X_train, Y_train, X_test, Y_test, factor, indice, X_forecast = transform_data(
        target_curve_name,
        canton,
        df,
        look_back=look_back,
        predict_n=predict_n,
        split=split,
    )

    return X_train, Y_train, X_test, Y_test, factor, indice, X_forecast


def transform_data(
    target_curve_name,
    canton,
    df,
    look_back=21,
    predict_n=14,
    split=0.75,
):
    indice = list(df.index)
    indice = [i.date() for i in indice]

    target_col = list(df.columns).index(f"{target_curve_name}_{canton}")

    norm_data, max_features = normalize_data(df)
    factor = max_features[target_col]

    X_forecast = np.empty((1, look_back, norm_data.shape[1]))

    X_forecast[:, :, :] = norm_data[-look_back:]

    X_train, Y_train, X_test, Y_test = split_data(
        norm_data, look_back, split, predict_n, Y_column=target_col
    )

    return X_train, Y_train, X_test, Y_test, factor, indice, X_forecast


def train_eval_canton(
    model,
    X_train,
    Y_train,
    X_test,
    Y_test,
    factor,
    indice,
    batch=1,
    epochs=100,
    path=None,
    label="region_name",
    uncertainty=True,
    save=False,
):

    history = train(
        model, X_train, Y_train, batch_size=batch, epochs=epochs, label=label, save=save
    )

    if save:
        if path is None:
            model.save(f"train_eval_lstm_{label}_epochs_{epochs}.h5")

        else:
            model.save(f"{path}/train_eval_lstm_{label}_epochs_{epochs}.h5")

    Y_data = np.concatenate((Y_train, Y_test), axis=0)  # type: ignore

    predicted_out = predict(model, X_test, Y_test, uncertainty)

    predicted_in = predict(model, X_train, Y_train, uncertainty)

    predicted = np.concatenate((predicted_in, predicted_out), axis=0)  # type: ignore

    df_pred = pd.DataFrame()

    if uncertainty:
        df_predicted = pd.DataFrame(np.percentile(predicted, 50, axis=2))
        df_predicted25 = pd.DataFrame(np.percentile(predicted, 2.5, axis=2))
        df_predicted975 = pd.DataFrame(np.percentile(predicted, 97.5, axis=2))

        df_pred["date"] = indice[X_train.shape[1] + Y_train.shape[1] :]

        df_pred["target"] = Y_data[1:, -1] * factor

        df_pred["lower"] = df_predicted25[df_predicted25.columns[-1]] * factor

        df_pred["median"] = df_predicted[df_predicted.columns[-1]] * factor

        df_pred["upper"] = df_predicted975[df_predicted975.columns[-1]] * factor

        df_pred["train_size"] = [
            len(X_train) - (X_train.shape[1] + Y_train.shape[1])
        ] * len(df_pred)

    else:
        if len(predicted.shape) == 2:
            df_predicted = pd.DataFrame(predicted).T

        df_pred["date"] = indice[Y_train.shape[1] + X_train.shape[1] :]

        df_pred["target"] = Y_data[:, -1] * factor

        df_pred["predict"] = df_predicted[df_predicted.columns[-1]] * factor

        df_pred["train_size"] = [
            len(X_train) - (Y_train.shape[1] + X_train.shape[1])
        ] * len(df_pred)

    return df_pred


def training_canton(
    model,
    X_train,
    Y_train,
    batch=1,
    epochs=100,
    path=None,
    label="region_name",
    save=True,
):

    history = train(
        model, X_train, Y_train, batch_size=batch, epochs=epochs, label=label, save=save
    )

    if path is None:
        model.save(f"trained_model_{label}_epochs_{epochs}.h5")
    else:
        model.save(f"{path}/trained_model_{label}_epochs_{epochs}.h5")

    return


def forecasting_canton(
    label, epochs, X_for, factor, indice, path=None, uncertainty=True
):

    if path is None:
        model = keras.models.load_model(f"trained_model_{label}_epochs_{epochs}.h5")
    else:
        model = keras.models.load_model(
            f"{path}/trained_model_{label}_epochs_{epochs}.h5"
        )

    if uncertainty:
        predicted = np.stack([model.predict(X_for, batch_size=1, verbose=1) for i in range(100)], axis=2)  # type: ignore
    else:
        predicted = model.predict(X_for, batch_size=1, verbose=1)

    forecast_dates = []

    last_day = datetime.strftime(indice[-1], "%Y-%m-%d")

    a = datetime.strptime(last_day, "%Y-%m-%d")

    for i in np.arange(1, len(predicted[0]) + 1):

        d_i = datetime.strftime(a + timedelta(days=int(i)), "%Y-%m-%d")

        forecast_dates.append(datetime.strptime(d_i, "%Y-%m-%d"))

    df_for = pd.DataFrame()

    if uncertainty:
        df_predicted = pd.DataFrame(np.percentile(predicted, 50, axis=2)).T
        df_predicted25 = pd.DataFrame(np.percentile(predicted, 2.5, axis=2)).T
        df_predicted975 = pd.DataFrame(np.percentile(predicted, 97.5, axis=2)).T

        df_for["date"] = forecast_dates

        df_for["lower"] = df_predicted25[df_predicted25.columns[-1]] * factor

        df_for["median"] = df_predicted[df_predicted.columns[-1]] * factor

        df_for["upper"] = df_predicted975[df_predicted975.columns[-1]] * factor

    else:
        if len(predicted.shape) == 2:
            df_predicted = pd.DataFrame(predicted).T

        df_for["date"] = forecast_dates

        df_for["predict"] = df_predicted[df_predicted.columns[-1]] * factor

    return df_for


# params_model = {'hidden': 12, 'epochs': 100,  'look_back' : 21,'predict_n': 14 }


def train_eval_single_canton(
    target_curve_name,
    canton,
    predictors,
    split=0.75,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    updated_data=True,
    uncertainity=True,
    save=False,
    hidden=12,
    epochs=100,
    look_back=21,
    predict_n=14,
):

    """
    Function to train and evaluate the model for one georegion

    Important:
    * By default the function is using the clustering cantons and the data since 2020
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    params canton: canton of interest
    params predictors: variables that  will be used in model
    params vaccine: It determines if the vaccine data from owid will be used or not
    params smooth: It determines if data will be smoothed or not
    params ini_date: Determines the beggining of the train dataset
    params title: If none the title will be: Hospitalizations - canton
    params path: If none the plot will be save in the directory: images/hosp_{canton}
    """

    # compute the clusters
    # print('cluster')
    # start = time.time()
    clusters = compute_clusters(
        "switzerland",
        "cases",
        ["datum", '"geoRegion"', "entries"],
        t=0.3,
        drop_georegions=["CH", "FL", "CHFL"],
        plot=False,
    )[1]
    # end = time.time()
    # print(end - start)

    for cluster in clusters:

        if canton in cluster:

            cluster_canton = cluster

    # getting the data
    # print(cluster_canton)
    # print('get_data')
    # start = time.time()

    X_train, Y_train, X_test, Y_test, factor, indice, X_forecast = get_data_model(
        target_curve_name,
        canton,
        cluster_canton,
        predictors,
        look_back=look_back,
        predict_n=predict_n,
        split=split,
        vaccine=vaccine,
        smooth=smooth,
        updated_data=updated_data,
    )

    model = build_model(
        hidden, X_train.shape[2], predict_n=predict_n, look_back=look_back
    )

    # get predictions
    df = train_eval_canton(
        model,
        X_train,
        Y_train,
        X_test,
        Y_test,
        factor,
        indice,
        batch=1,
        epochs=epochs,
        label=f"{target_curve_name}_{canton}",
        uncertainty=uncertainity,
        save=save,
    )

    return df


def train_eval_all_cantons(
    target_curve_name,
    predictors,
    split=0.75,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    updated_data=True,
    uncertainity=True,
    save=False,
    hidden=12,
    epochs=100,
    look_back=21,
    predict_n=14,
):

    """
    Function to make prediction for all the cantons

    Important:
    * By default the function is using the clustering cantons and the data since 2020
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    params target_curve_name: string to indicate the target column of the predictions
    params predictors: variables that  will be used in model
    params vaccine: It determines if the vaccine data from owid will be used or not
    params smooth: It determines if data will be smoothed or not
    params ini_date: Determines the beggining of the train dataset

    returns: Dataframe with the predictions for all the cantons
    """

    df_all = pd.DataFrame()

    # compute the clusters
    clusters = compute_clusters(
        "switzerland",
        "cases",
        ["datum", '"geoRegion"', "entries"],
        t=0.3,
        drop_georegions=["CH", "FL", "CHFL"],
        plot=False,
    )[1]

    for cluster in clusters:
        # getting the data
        df = get_cluster_data(predictors, list(cluster), vaccine=vaccine, smooth=smooth)
        # filling the nan values with 0
        df = df.fillna(0)

        for canton in cluster:
            # apply the model

            (
                X_train,
                Y_train,
                X_test,
                Y_test,
                factor,
                indice,
                X_forecast,
            ) = transform_data(
                target_curve_name,
                canton,
                df,
                look_back=look_back,
                predict_n=predict_n,
                split=split,
            )
            model = build_model(
                hidden, X_train.shape[2], predict_n=predict_n, look_back=look_back
            )

            # get predictions
            df_pred = train_eval_canton(
                model,
                X_train,
                Y_train,
                X_test,
                Y_test,
                factor,
                indice,
                batch=1,
                epochs=epochs,
                label=f"{target_curve_name}_{canton}",
                uncertainty=uncertainity,
                save=save,
            )

            df_all = pd.concat([df_all, df_pred])

    return df_all


def train_single_canton(
    target_curve_name,
    canton,
    predictors,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    updated_data=True,
    uncertainity=True,
    save=False,
    path=None,
    hidden=12,
    epochs=100,
    look_back=21,
    predict_n=14,
):

    """
    Function to train and evaluate the model for one georegion

    Important:
    * By default the function is using the clustering cantons and the data since 2020
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    params canton: canton of interest
    params predictors: variables that  will be used in model
    params vaccine: It determines if the vaccine data from owid will be used or not
    params smooth: It determines if data will be smoothed or not
    params ini_date: Determines the beggining of the train dataset
    params title: If none the title will be: Hospitalizations - canton
    params path: If none the plot will be save in the directory: images/hosp_{canton}
    """

    # compute the clusters
    clusters = compute_clusters(
        "switzerland",
        "cases",
        ["datum", '"geoRegion"', "entries"],
        t=0.3,
        drop_georegions=["CH", "FL", "CHFL"],
        plot=False,
    )[1]

    for cluster in clusters:

        if canton in cluster:

            cluster_canton = cluster

    X_train, Y_train, X_test, Y_test, factor, indice, X_forecast = get_data_model(
        target_curve_name,
        canton,
        cluster_canton,
        predictors,
        look_back=look_back,
        predict_n=predict_n,
        split=1,
        vaccine=vaccine,
        smooth=smooth,
        updated_data=updated_data,
    )

    model = build_model(
        hidden, X_train.shape[2], predict_n=predict_n, look_back=look_back
    )

    training_canton(
        model,
        X_train,
        Y_train,
        batch=1,
        epochs=epochs,
        path=path,
        label=canton,
        save=True,
    )

    return


def train_all_cantons(
    target_curve_name,
    canton,
    predictors,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    updated_data=True,
    uncertainity=True,
    save=False,
    path=None,
    hidden=12,
    epochs=100,
    look_back=21,
    predict_n=14,
):
    """

    Important:
    * By default the function is using the clustering cantons and the data since 2020
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    params target_curve_name: string to indicate the target column of the predictions
    params predictors: variables that  will be used in model
    params vaccine: It determines if the vaccine data from owid will be used or not
    params smooth: It determines if data will be smoothed or not
    params ini_date: Determines the beggining of the train dataset

    returns: Dataframe with the forecast for all the cantons
    """

    # compute the clusters
    clusters = compute_clusters(
        "switzerland",
        "cases",
        ["datum", '"geoRegion"', "entries"],
        t=0.3,
        drop_georegions=["CH", "FL", "CHFL"],
        plot=False,
    )[1]

    for cluster in clusters:

        df = get_cluster_data(predictors, list(cluster), vaccine=vaccine, smooth=smooth)
        # filling the nan values with 0
        df = df.fillna(0)

        for canton in cluster:

            (
                X_train,
                Y_train,
                X_test,
                Y_test,
                factor,
                indice,
                X_forecast,
            ) = transform_data(
                target_curve_name,
                canton,
                df,
                look_back=look_back,
                predict_n=predict_n,
                split=1,
            )

            # train the models and save in the server
            model = build_model(
                hidden, X_train.shape[2], predict_n=predict_n, look_back=look_back
            )

            training_canton(
                model,
                X_train,
                Y_train,
                batch=1,
                epochs=epochs,
                path=path,
                label=canton,
                save=True,
            )

    return


def forecast_single_canton(
    label,
    epochs,
    target_curve_name,
    canton,
    predictors,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    updated_data=True,
    uncertainity=True,
    save=False,
    path=None,
    look_back=21,
    predict_n=14,
):

    # compute the clusters
    clusters = compute_clusters(
        "switzerland",
        "cases",
        ["datum", '"geoRegion"', "entries"],
        t=0.3,
        drop_georegions=["CH", "FL", "CHFL"],
        plot=False,
    )[1]

    for cluster in clusters:

        if canton in cluster:

            cluster_canton = cluster

    X_train, Y_train, X_test, Y_test, factor, indice, X_forecast = get_data_model(
        target_curve_name,
        canton,
        cluster_canton,
        predictors,
        look_back=look_back,
        predict_n=predict_n,
        split=1,
        vaccine=vaccine,
        smooth=smooth,
        updated_data=updated_data,
    )

    df_for = forecasting_canton(
        label, epochs, X_forecast, factor, indice, path=None, uncertainty=True
    )

    return df_for


def forecast_all_cantons(
    target_curve_name,
    canton,
    predictors,
    epochs=100,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    updated_data=True,
    uncertainity=True,
    save=False,
    path=None,
    look_back=21,
    predict_n=14,
):
    """
    Function to make the forecast for all the cantons

    Important:
    * By default the function is using the clustering cantons and the data since 2020
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    params target_curve_name: string to indicate the target column of the predictions
    params predictors: variables that  will be used in model
    params vaccine: It determines if the vaccine data from owid will be used or not
    params smooth: It determines if data will be smoothed or not
    params ini_date: Determines the beggining of the train dataset

    returns: Dataframe with the forecast for all the cantons
    """
    df_all = pd.DataFrame()

    # compute the clusters
    clusters = compute_clusters(
        "switzerland",
        "cases",
        ["datum", '"geoRegion"', "entries"],
        t=0.3,
        drop_georegions=["CH", "FL", "CHFL"],
        plot=False,
    )[1]

    for cluster in clusters:

        df = get_cluster_data(predictors, list(cluster), vaccine=vaccine, smooth=smooth)
        # filling the nan values with 0
        df = df.fillna(0)

        for canton in cluster:

            # apply the model

            (
                X_train,
                Y_train,
                X_test,
                Y_test,
                factor,
                indice,
                X_forecast,
            ) = transform_data(
                target_curve_name,
                canton,
                df,
                look_back=look_back,
                predict_n=predict_n,
                split=1,
            )

            # get predictions and forecast
            df_for = forecasting_canton(
                canton, epochs, X_forecast, factor, indice, path=None, uncertainty=True
            )

            df_all = pd.concat([df_all, df_for])

    return df_all
