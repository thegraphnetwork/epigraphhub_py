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
                        history of the model. The file is named following: `history_{label}.pkl`. The 
                        file with the model are saved as `{label}_{epochs}.h5`. 


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
            model.save(f"{label}_{epochs}.h5")
            with open(f"history_{label}.pkl", "wb") as f:
                pickle.dump(hist.history, f)
        else:
            model.save(f"{path}/{label}_{epochs}.h5")
            with open(f"{path}/history_{label}.pkl", "wb") as f:
                pickle.dump(hist.history, f)

    return hist, model 


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

def train_eval(
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
    label="train_eval_region_name",
    uncertainty=True,
    save=False,
):
    """
    Function to train an evaluate the model for a single canton

    :params model: compiled LSTM model. 
    :params X_train: array. Array with the features to train the model.
    :params Y_train: array. Array with the targets to train the model.
    :params X_test: array. Array with the features to test the model.
    :params Y_test: array. Array with the targets to test the model.
    :params factor: array. Array with the weights used to normalize the data.
    :params indice: array. Array with the dates associated with the arrays in 
                            X_train and X_test 
    :params batch: int. batch size for batch training
    :params epoch: int. Number of epochs to train the model
    :params uncertainty: boolean. Indicates if the confidence intervals will be computed or not 
    :params save: boolean. Indicates if the trained model will be saved or not
    :params path: string. Path where the model will be saved 
    :params label: string. Name used to save the model 

    :returns: dataframe
    """

    history, model = train(
        model, X_train, Y_train, batch_size=batch, epochs=epochs, label=label, save=save, path = path)

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

    else:
        if len(predicted.shape) == 2:
            df_predicted = pd.DataFrame(predicted).T

        df_pred["date"] = indice[Y_train.shape[1] + X_train.shape[1] :]

        df_pred["target"] = Y_data[:, -1] * factor

        df_pred["predict"] = df_predicted[df_predicted.columns[-1]] * factor

        df_pred["train_size"] = len(X_train) - (Y_train.shape[1] + X_train.shape[1])

    return df_pred

def train_eval_single_canton(
    target_curve_name,
    canton,
    predictors,
    cantons, 
    vaccine,
    smooth,
    ini_date, 
    split,
    look_back, 
    predict_n,
    batch=1,
    hidden = 8, 
    epochs=100,
    uncertainty = True,  
    save = True, 
    path=None):

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
    :params look_back: int. Number of the last days that will be used to forecast the next days
    :params predict_n: int. Number of days that will be predicted
    :param batch_size: batch size for batch training
    :param hidden: number of hidden nodes
    :params epochs: int. Number of epochs to train the model
    :params uncertainty: boolean. Indicates if the confidence intervals will be computed or not 
    :params save: boolean. Indicates if the trained model will be saved or not
    :params path: string. Path where the model will be saved 

    returns: Dataframe.
    """
    X_train, Y_train, X_test, Y_test, factor, indice, X_forecast = get_data_model(
        target_curve_name,
        canton,
        cantons,
        predictors,
        look_back=look_back,
        predict_n=predict_n,
        split=split,
        vaccine=vaccine,
        smooth=smooth,
        ini_date = ini_date)

    model = build_model(
        hidden, X_train.shape[2], predict_n=predict_n, look_back=look_back
    )

    df_pred = train_eval(
        model,
        X_train,
        Y_train,
        X_test,
        Y_test,
        factor,
        indice,
        batch=batch,
        epochs=epochs,
        path=path,
        label=f'train_eval_lstm_{target_curve_name}_{canton}_{epochs}',
        uncertainty=uncertainty,
        save=save)

    df_pred['canton'] = canton

    return df_pred


def train_eval_all_cantons(
    target_curve_name,
    canton,
    predictors, 
    vaccine,
    smooth,
    ini_date, 
    split,
    look_back, 
    predict_n,
    batch=1,
    hidden = 8, 
    epochs=100,
    uncertainty = True,  
    save = True, 
    path=None):

    """
    Function to train and evaluate the model for one georegion.

    Important:
    * By default in the function, for each canton is used your own data as predictors
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients

    :params target_curve_name: string to indicate the target column of the predictions
    :params canton: canton of interest
    :params predictors: variables that  will be used in model
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed (7 day moving average) or not
    :params ini_date: Determines the beggining of the train dataset
    :params split: float. Determines which percentage of the data will be used to train the model
    :params look_back: int. Number of the last days that will be used to forecast the next days
    :params predict_n: int. Number of days that will be predicted
    :param batch_size: batch size for batch training
    :param hidden: number of hidden nodes
    :params epochs: int. Number of epochs to train the model
    :params uncertainty: boolean. Indicates if the confidence intervals will be computed or not 
    :params save: boolean. Indicates if the trained model will be saved or not
    :params path: string. Path where the model will be saved 

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

        df_pred = train_eval_single_canton(
                        target_curve_name,
                        canton,
                        predictors,
                        [canton],
                        vaccine = vaccine,
                        smooth = smooth,
                        ini_date = ini_date, 
                        split = split,
                        look_back = look_back, 
                        predict_n = predict_n,
                        batch = batch,
                        hidden = hidden, 
                        epochs = epochs,
                        uncertainty = uncertainty,  
                        save = save, 
                        path = path)

        df_all = pd.concat([df_all, df_pred])

    return df_all


def train_single_canton(
    target_curve_name,
    canton,
    predictors,
    cantons, 
    vaccine,
    smooth,
    ini_date, 
    look_back, 
    predict_n,
    batch=1,
    hidden = 8, 
    epochs=100, 
    save = True, 
    path=None):

    """
    Function to train the model for one georegion.

    Important:
    * For the predictor hospCapacity is used as predictor the column ICU_Covid19Patients
    :params target_curve_name: string to indicate the target column of the predictions
    :params canton: canton of interest
    :params predictors: variables that  will be used in model
    :params cantons: list of other cantons whose data will be used as predictors
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed (7 day moving average) or not
    :params ini_date: Determines the beggining of the train dataset
    :params look_back: int. Number of the last days that will be used to forecast the next days
    :params predict_n: int. Number of days that will be predicted
    :param batch: batch size for batch training
    :param hidden: number of hidden nodes
    :params epochs: int. Number of epochs to train the model
    :params save: boolean. Indicates if the trained model will be saved or not
    :params path: string. Path where the model will be saved 
    
    returns: model, history 
    """

    X_train, Y_train, X_test, Y_test, factor, indice, X_forecast = get_data_model(
        target_curve_name,
        canton,
        cantons,
        predictors,
        look_back=look_back,
        predict_n=predict_n,
        split=1,
        vaccine=vaccine,
        smooth=smooth,
        ini_date = ini_date 
    )
    
    model = build_model(
        hidden, X_train.shape[2], predict_n=predict_n, look_back=look_back
    )

    history, model = train(
        model, X_train, Y_train, batch_size=batch, epochs=epochs, label=f'train_lstm_{target_curve_name}_{canton}_{epochs}', save=save, path = path)

    return model, history 

def train_all_cantons(
    target_curve_name,
    canton,
    predictors,
    vaccine,
    smooth,
    ini_date, 
    look_back, 
    predict_n,
    batch=1,
    hidden = 8, 
    epochs=100, 
    save = True, 
    path=None):

    """
    Function to train the model for one georegion.

    Important:
    * By default in the function, for each canton is used your own data as predictors

    :params target_curve_name: string to indicate the target column of the predictions
    :params canton: canton of interest
    :params predictors: variables that  will be used in model
    :params cantons: list of other cantons whose data will be used as predictors
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed (7 day moving average) or not
    :params ini_date: Determines the beggining of the train dataset
    :params look_back: int. Number of the last days that will be used to forecast the next days
    :params predict_n: int. Number of days that will be predicted
    :param batch: batch size for batch training
    :param hidden: number of hidden nodes
    :params epochs: int. Number of epochs to train the model
    :params save: boolean. Indicates if the trained model will be saved or not
    :params path: string. Path where the model will be saved 
    
    
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

        train_single_canton(
                        target_curve_name,
                        canton,
                        predictors,
                        [canton], 
                        vaccine=vaccine,
                        smooth =smooth,
                        ini_date = ini_date, 
                        look_back = look_back, 
                        predict_n = predict_n,
                        batch = batch,
                        hidden = hidden, 
                        epochs = epochs, 
                        save = save, 
                        path = path)


    

    return 



def forecast(
    label,path,  X_for, factor, indice, uncertainty=True
):
    """
    Function to apply the forecast using a pre trained model.

    Important:
    :params label: string. Name used to save the model 
    :params path: string. Path where the model is saved
    :params X_for: array. Array that will be used to forecast the data 
    :params factor: array. It will be used to transform the predictions to the right scale
    :params indice: array | list. It's the date associated with X_for.  
    :params uncertainty: boolean. If true the confidence interval of the predicitions will be computed. 

    :return dataframe. 
    """



    if path is None:
        model = keras.models.load_model(f"{label}.h5")
    else:
        model = keras.models.load_model(
            f"{path}/{label}.h5"
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


def forecast_single_canton(
    target_curve_name,
    canton,
    predictors,
    cantons,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    look_back=21,
    predict_n=14,
    path = None,
    epochs = 300,
    uncertainty=True,
):
    """
    Function to load a trained model and use it to forecast one canton.

    Important:
    :params target_curve_name: string to indicate the target column of the predictions
    :params canton: canton of interest
    :params predictors: variables that  will be used in model
    :params cantons: list of other cantons whose data will be used as predictors
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed (7 day moving average) or not
    :params ini_date: Determines the beggining of the train dataset
    :params look_back: int. Number of the last days that will be used to forecast the next days
    :params predict_n: int. Number of days that will be predicted
    :params path: string. Path where the trained model was saved
    :params epochs: int. Number of epochs used to train the model
    :params uncertainty: boolean. If true the confidence interval of the predicitions will be computed.   
    
    :return DataFrame 
    """


    X_train, Y_train, X_test, Y_test, factor, indice, X_forecast = get_data_model(
        target_curve_name,
        canton,
        cantons,
        predictors,
        look_back=look_back,
        predict_n=predict_n,
        split=1,
        vaccine=vaccine,
        smooth=smooth,
        ini_date = ini_date
    )

    df_for = forecast(f'train_lstm_{target_curve_name}_{canton}_{epochs}',path, X_forecast, factor, indice, uncertainty = uncertainty)

    df_for['canton'] = canton 

    return df_for


def forecast_all_cantons(
    target_curve_name,
    canton,
    predictors,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    look_back=21,
    predict_n=14,
    path = None,
    epochs = 300,
    uncertainty=True,
):
    """
    Function to load a trained model and use it to forecast all canton.
    Important:
    * By default in the function, for each canton is used your own data as predictors

    :params target_curve_name: string to indicate the target column of the predictions
    :params canton: canton of interest
    :params predictors: variables that  will be used in model
    :params vaccine: It determines if the vaccine data from owid will be used or not
    :params smooth: It determines if data will be smoothed (7 day moving average) or not
    :params ini_date: Determines the beggining of the train dataset
    :params look_back: int. Number of the last days that will be used to forecast the next days
    :params predict_n: int. Number of days that will be predicted
    :params path: string. Path where the trained model was saved
    :params epochs: int. Number of epochs used to train the model
    :params uncertainty: boolean. If true the confidence interval of the predicitions will be computed.   
    
    :return DataFrame 
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

        df_for = forecast_single_canton(
                            target_curve_name,
                            canton,
                            predictors,
                            [canton],
                            vaccine=True,
                            smooth=True,
                            ini_date="2020-03-01",
                            look_back=21,
                            predict_n=14,
                            path = None,
                            epochs = 300,
                            uncertainty=True)

        df_all = pd.concat([df_all, df_for])

    return df_all


def plot_predicted_vs_data(
    predicted,
    Ydata,
    indice,
    label, 
    factor,
    split_point=None,
    uncertainty=False,
    save = True, 
    path = None, 
    label_save = 'predict_vs_data'
):
    """
    Plot  only the last (furthest) prediction points against data

    :params predicted: array. model predictions
    :params Ydata: array. observed data
    :params indice: array|list. Dates associated with the predicted and Ydata. 
    :params label: string. Name associated with the predictions. It has this format: `Predictions for {label}`
    :params factor: array. Normalizing factor for the target variable
    :params split_point: int. Location of the value that split the Ydata in train and test.
    :params uncertainty: boolean. If true the confidence interval are predicted 
    :params save: boolean. If true the figure created is saved
    :params path: string|None. Path where save the figure 
    :params label_save: string. Name to save the figure 
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
    if (uncertainty) and (len(predicted.shape) != 2):
        plt.fill_between(
            indice[len(indice) - Ydata.shape[0] :],
            df_predicted25[df_predicted25.columns[-1]] * factor,
            df_predicted975[df_predicted975.columns[-1]] * factor,
            color="b",
            alpha=0.3,
        )

    plt.grid()
    plt.title(f"Predictions for {label}")
    plt.xlabel("time")
    plt.ylabel("incidence")
    plt.xticks(rotation=70)
    plt.legend(["data", "predicted"])

    if save:
        if path == None:
            plt.savefig(f"{label_save.png}", bbox_inches="tight", dpi=300)
        else: 
            plt.savefig(f"{path}/{label_save}.png", bbox_inches="tight", dpi=300)

    plt.show()
    
    return 
