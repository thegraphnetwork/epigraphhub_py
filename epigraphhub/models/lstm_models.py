"""
The functions in this module allow the application of a
LSTM (Long short-term memory) model for time series. There are separate 
functions to train and evaluate (separate the data in train and test 
datasets), train with all the data available, and make
forecasts. Also, there are functions focused in apply this model 
in the Switzerland data for just one 
canton or all the cantons of Switzerland. 
"""

import pickle
from datetime import datetime, timedelta
from time import time

import numpy as np
import pandas as pd
import tensorflow.keras as keras
from matplotlib import pyplot as plt
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

from epigraphhub.data.get_data import get_cluster_data
from epigraphhub.data.preprocessing import lstm_split_data as split_data
from epigraphhub.data.preprocessing import normalize_data


def build_model(hidden, features, predict_n, look_back=10, batch_size=1):
    """
    Builds and returns the LSTM model with the parameters given
    :param hidden: number of hidden nodes
    :param features: number of variables in the example table
    :param predict_n: number of days that will be predicted
    :param look_back: Number of time-steps to look back before predicting
    :param batch_size: batch size for batch training
    :return:
    """

    inp = keras.Input(
        shape=(look_back, features),
        # batch_shape=(batch_size, look_back, features)
    )
    x = LSTM(
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

    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        kernel_initializer="he_uniform",
        stateful=False,
        batch_input_shape=(batch_size, look_back, features),
        activation="relu",
        dropout=0.1,
        recurrent_dropout=0.1,
        implementation=2,
        unit_forget_bias=True,
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
    model.compile(
        loss="msle", optimizer="adam", metrics=["accuracy", "mape", "mse", "msle"]
    )
    print("Compilation Time : ", time() - start)
    plot_model(model, to_file="LSTM_model.png")
    print(model.summary())
    return model


def train(
    model, X_train, Y_train, batch_size=1, epochs=10, save=False, path=None, label="GE"
):
    """
    traia a LSTM model
    :param model: LSTM model
    :param X_train: array with the features to train the model
    :param Y_train: array with the targets to train the model
    :param batch_size: batch size for batch training
    :param epochs: int. Number of epochs to train the model
    :param save: boolean. Indicates if the history of the model will be save or not
    :param path: string. Indicates the path where the history of the model will be saved
    :param label: string. String used to name the .pkl file with the
                        history of the model. The file is named following: `history_{label}.pkl`


    :return: model trained
    """

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
            with open(f"history_{label}.pkl", "wb") as f:
                pickle.dump(hist.history, f)
        else:
            with open(f"{path}/history_{label}.pkl", "wb") as f:
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


def predict(model, Xdata, uncertainty=False):
    """'
    Funtion used to make the predictions given a trained model and a array with
    the same size of the array used to train the model.
    :param model: trained model.
    :param Xdata: array.
    :param uncertainty: boolean. If true, it's returned 100 different predictions that
                        can be used to compute the confidence interval of the predictions.

    :returns: array.
    """
    if uncertainty:
        predicted = np.stack([model.predict(Xdata, batch_size=1, verbose=1) for i in range(100)], axis=2)  # type: ignore
    else:
        predicted = model.predict(Xdata, batch_size=1, verbose=1)

    return predicted


def calculate_metrics(pred, ytrue, factor):
    """
    Function to return a dataframe with some error metrics computed.

    :params pred: array.
    :params ytrue: array.
    :params factor: array.
    :returns: dataframe.
    """
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
    ini_date="2020-08-01",
):
    """
    Function specific to transform the swiss data in the right format.

    :params target_curve_name: String. string to indicate the target column of the predictions
    :params canton: string. canton of interest
    :params cluster: list of strings.  list of other cantons whose data will be used as predictors
    :params predictors: list of strings. variables that  will be used in model
    :params lookback: int. Number of the last days that will be used to forecast the next days
    :params predict_n: int. Number of days that will be predicted
    :params split: float (0,1). Determines which percentage of the data will be used to train the model
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed (7 day moving average) or not
    :params ini_date: string. Determines the beggining of the train dataset

    :return: arrays
    """
    df = get_cluster_data(
        "switzerland", predictors, cluster, vaccine=vaccine, smooth=smooth
    )

    df = df.loc[ini_date:]

    if target_curve_name != "hosp":

        for i in df.columns:

            if i.startswith("diff") and (
                i.endswith(f"{target_curve_name}_{canton}") == False
            ):

                del df[i]

    df = df.fillna(0)

    X_train, Y_train, X_test, Y_test, factor, indice, X_forecast = transform_data(
        f"{target_curve_name}_{canton}",
        df,
        look_back=look_back,
        predict_n=predict_n,
        split=split,
    )

    return X_train, Y_train, X_test, Y_test, factor, indice, X_forecast


def transform_data(
    target_name,
    df,
    look_back=21,
    predict_n=14,
    split=0.75,
):
    """
    Function to transform a data frame with datetime index and features and target
    values in the columns in the format accepted by a neural network model.

    :params target_name: string. String with the name of the target column
    :params df: dataframe. Dataframe with features and target column.
    :params lookback: int. Number of the last days that will be used to forecast the next days
    :params predict_n: int. Number of days that will be predicted
    :params split: float (0,1). Determines which percentage of the data will be used to train the model

    :returns: arrays.
    """

    indice = list(df.index)
    indice = [i.date() for i in indice]

    target_col = list(df.columns).index(target_name)

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
    """
    Function to train an evaluate the model for a single canton

    :params model:
    :params X_train: array.
    :params Y_train: array.
    :params X_test: array.
    :params Y_test: array.
    :params factor: array.
    :params indice: array.
    :params batch: int.
    :params epoch: int.
    :params uncertainty: boolean.
    :params save: boolean.
    :params path: string.
    :params label: string.

    :returns: dataframe

    """

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

        df_pred["train_size"] = len(X_train) - (X_train.shape[1] + Y_train.shape[1])

        df_pred["canton"] = label

    else:
        if len(predicted.shape) == 2:
            df_predicted = pd.DataFrame(predicted).T

        df_pred["date"] = indice[Y_train.shape[1] + X_train.shape[1] :]

        df_pred["target"] = Y_data[:, -1] * factor

        df_pred["predict"] = df_predicted[df_predicted.columns[-1]] * factor

        df_pred["train_size"] = len(X_train) - (Y_train.shape[1] + X_train.shape[1])

        df_pred["canton"] = label

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

        df_for["canton"] = label[-2:]
    else:
        if len(predicted.shape) == 2:
            df_predicted = pd.DataFrame(predicted).T

        df_for["date"] = forecast_dates

        df_for["predict"] = df_predicted[df_predicted.columns[-1]] * factor

        df_for["canton"] = label[-2:]

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
    uncertainty=True,
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

    # getting the data
    # print(cluster_canton)
    # print('get_data')
    # start = time.time()

    X_train, Y_train, X_test, Y_test, factor, indice, X_forecast = get_data_model(
        target_curve_name,
        canton,
        [canton],
        predictors,
        look_back=look_back,
        predict_n=predict_n,
        split=split,
        vaccine=vaccine,
        smooth=smooth,
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
        uncertainty=uncertainty,
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

        df = get_cluster_data(predictors, list(canton), vaccine=vaccine, smooth=smooth)
        df = df.fillna(0)

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

    X_train, Y_train, X_test, Y_test, factor, indice, X_forecast = get_data_model(
        target_curve_name,
        canton,
        [canton],
        predictors,
        look_back=look_back,
        predict_n=predict_n,
        split=1,
        vaccine=vaccine,
        smooth=smooth,
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

        df = get_cluster_data(predictors, list(canton), vaccine=vaccine, smooth=smooth)
        # filling the nan values with 0
        df = df.fillna(0)

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
    uncertainity=True,
    save=False,
    path=None,
    look_back=21,
    predict_n=14,
):

    X_train, Y_train, X_test, Y_test, factor, indice, X_forecast = get_data_model(
        target_curve_name,
        canton,
        [canton],
        predictors,
        look_back=look_back,
        predict_n=predict_n,
        split=1,
        vaccine=vaccine,
        smooth=smooth,
    )

    df_for = forecasting_canton(
        label, epochs, X_forecast, factor, indice, path=path, uncertainty=True
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

        df = get_cluster_data(predictors, list(canton), vaccine=vaccine, smooth=smooth)
        # filling the nan values with 0
        df = df.fillna(0)

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


def plot_predicted_vs_data(
    predicted,
    Ydata,
    indice,
    canton,
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
