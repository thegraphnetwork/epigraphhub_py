"""
The functions in this module allow the application of any scikit-learn regressor model that have
the methods .fit and .predict implemented. 
There are separate functions to train and evaluate (separate the data in train and test datasets),
train with all the data available, and make forecasts. Also, there is the possibilite to transform 
the data using methods like sklearn.preprocessing.MinMaxScaler and save the models trained. 
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
    model, idx_preds, X_train, X_test, targets, horizon_forecast=14, 
    norm = False, norm_model_x =  MinMaxScaler(), norm_model_y =  MinMaxScaler(), 
    save = False, path = None, label= 'test'):
    '''
    This function train {horizon_forecast} models (one for each time series forecast) with the X_train
    datasets and targets dictionaty and test the model with the X_test dataset. When we train the model
    with all the data, that's X_test = None the function return a empty dataframe. 
    params model: class of model that you to apply in teh data. 
    params idx: list of dates used to associate the prediction made with the date of prediction 
    params X_train: dataframe to train the models (with columns with lagged values)
    params X_test: dataframe to test the models (with columns with lagged values)
    params targets: dictionary with the targets for each forecast time step 
    params horizon_forecast: number of forecast time steps that will be forecast
    param norm: decide if the data will be normalized 
    param norm_model_x: model to normalize the features (X_train and X_test datasets)
    param norm_model_y: model to normalize the targets (arrays in targets dict)
    params save: decides if the models will be save 
    params path: decides in which folder the models will be saved 
    params label: possibilite add a different name annotation for the  models saved. 
    By default the models are saved as:
    - trained_model_norm_{label}_{T}D.joblib' (when norm == True)
    - trained_model_{label}_{T}D.joblib' (when norm == False)
    and the norm_model_X: 
    - feature_norm_{label}.joblib (just applied and saved when norm == True)
    and the norm_model_y:
    - target_norm_{label}_{T}D.joblib (just applied and saved when norm == True)
    In the file names T represents each time step forecast. 
    returns: (
        df_pred - pandas DataFrame with the predictions, 
        models - dict with the trained models, 
        scx - model used to normalize X_train,  
        scy - dict with the models to normalize the targets)
    '''

    # if X_test = none it's not possible to make the predictions 
    if isinstance(X_test, pd.core.frame.DataFrame):
        preds = np.empty((len(idx_preds), horizon_forecast))

    if norm: 
        scx = norm_model_x
        X_train = scx.fit_transform(X_train)

        if save:
            if path is None:
                dump(scx, f'feature_norm_{label}.joblib')
            else:
                dump(scx, f'{path}feature_norm_{label}.joblib')
    else: 
        scx = None 

    models = {}
    dict_scy = {}
    for T in range(1, horizon_forecast + 1):

        if len(X_train) > len(targets[T]):
            X_train_t = X_train.iloc[: len(targets[T])]
            tgt = targets[T]

        else:
            X_train_t = X_train 
            tgt = targets[T][: len(X_train_t)]

        if norm:
            scy = norm_model_y
            tgt = scy.fit_transform(tgt.values.reshape(-1,1))
            model.fit(X_train_t, tgt.ravel())

            dict_scy[f"target_norm_{T}"] = scy

            if save: 
                if path is None:
                    dump(model, f'trained_model_norm_{label}_{T}D.joblib')
                    dump(scy, f'target_norm_{label}_{T}D.joblib')

                else:
                    dump(model, f'{path}trained_model_norm_{label}_{T}D.joblib')
                    dump(scy, f'{path}target_norm_{label}_{T}D.joblib') 

        else:
            model.fit(X_train_t, tgt)
            if path is None:
                dump(model, f'trained_model_{label}_{T}D.joblib')
            else:
                dump(model, f'{path}trained_model_{label}_{T}D.joblib')


        models[f"model_{T}"] = model

        if  isinstance(X_test, pd.core.frame.DataFrame):

            if norm:
                pred = model.predict(scx.transform(X_test.loc[idx_preds]))
                pred = np.array(scy.inverse_transform(pred.reshape(-1,1)))[:,0]

            else:
                pred = model.predict(X_test.loc[idx_preds])

            preds[:, (T - 1)] = pred

    # transformando preds em um array
    if isinstance(X_test, pd.core.frame.DataFrame):
        train_size = len(X_train)

        y = preds.flatten()

        x = pd.period_range(
            start=X_test.index[0], end=X_test.index[-1], freq="D"
        ).to_timestamp()

        x = np.array(x)

        y = np.array(y)

        target = targets[1]

        train_size = len(X_train)

        dif = len(x) - len(y)

        if dif < 0:
            y = y[: len(y) + dif]

        df_pred = pd.DataFrame()
        df_pred["target"] = target.values
        df_pred["date"] = x
        df_pred["predict"] = y
        df_pred["train_size"] = [train_size] * len(df_pred)
        df_pred.set_index('date', inplace = True)
        df_pred.index = pd.to_datetime(df_pred.index)

    if isinstance(X_test, pd.core.frame.DataFrame) == False:
        df_pred = pd.DataFrame()

    return df_pred, models, scx, dict_scy 

def rolling_forecast_mult(X_test, horizon_forecast=14, norm = False, path = None, label= 'test'):
    ''''
    Funtion to forecast the data. This function used pre saved models for this reason it's necessary 
    to provide the path, label, horizon_forecast and norm configurations used in the function rolling_predictions_mul
    params X_test: Dataframe (with colummns equal in the X_train used in the other function) and just
                   one line 
    params horizon_forecast: number of forecast time steps that will be forecast
    params path: folder to load the models 
    params label: label to load the models. 
    returns: dataframe with one column with the values forecasted and with the dates associated
    with the forecast in the index. 
    '''
    forecast_dates = []

    last_day = datetime.strftime((X_test.index)[-1], "%Y-%m-%d")

    a = datetime.strptime(last_day, "%Y-%m-%d")

    for i in np.arange(1, horizon_forecast + 1):

        d_i = datetime.strftime(a + timedelta(days=int(i)), "%Y-%m-%d")

        forecast_dates.append(datetime.strptime(d_i, "%Y-%m-%d"))

    forecasts = []

    if norm:
        if path is None:
            scx = load(f'feature_norm_{label}.joblib')
        else:
            scx = load(f'{path}feature_norm_{label}.joblib')
               

    for T in range(1, horizon_forecast + 1):

        if norm:
            if path is None:
                model = load(f'trained_model_norm_{label}_{T}D.joblib')
                scy = load(f'target_norm_{label}_{T}D.joblib')
            else:
                model = load(f'{path}trained_model_norm_{label}_{T}D.joblib')
                scy = load(f'{path}target_norm_{label}_{T}D.joblib')

        else:
            if path is None:
                model = load(f'trained_model_{label}_{T}D.joblib')
            else:
                model = load(f'{path}trained_model_{label}_{T}D.joblib')

        if norm:
            pred = model.predict(scx.transform(X_test))
            pred = np.array(scy.inverse_transform(pred.reshape(-1,1)))[:,0]
        else:
            pred = model.predict(X_test)

        forecasts.append(pred)

    df_for = pd.DataFrame()
    df_for["date"] = forecast_dates
    df_for["forecast"] = np.array(forecasts)
    df_for.set_index('date', inplace = True)
    df_for.index = pd.to_datetime(df_for.index)

    return df_for 


def rolling_predictions_one(
    model, idx_preds, X_train, X_test, targets, horizon_forecast=14,
    norm_model_x =  MinMaxScaler(), norm_model_y =  MinMaxScaler(),  norm = False,
    save = False, path = None, label= 'test'):
    '''
    This function train one model trained to predict the X + {horizon_forecast} day. 
     When we train the model with all the data, that's X_test = None the function return 
     a empty dataframe. 
    params model: class of model that you to apply in teh data. 
    params idx_preds: list of dates used to associate the prediction made with the date of prediction 
    params X_train: dataframe to train the models (with columns with lagged values)
    params X_test: dataframe to test the models (with columns with lagged values)
    params targets: dictionary with the targets for each forecast time step 
    params horizon_forecast: number of forecast time steps that will be forecast
    param norm: decide if the data will be normalized 
    param norm_model_x: model to normalize the features (X_train and X_test datasets)
    param norm_model_y: model to normalize the targets (arrays in targets dict)
    params save: decides if the models will be save 
    params path: decides in which folder the models will be saved 
    params label: possibilite add a different name annotation for the  models saved. 
    By default the models are saved as:
    - trained_model_norm_{label}_{T}D.joblib' (when norm == True)
    - trained_model_{label}_{T}D.joblib' (when norm == False)
    and the norm_model_X: 
    - feature_norm_{label}.joblib (just applied and saved when norm == True)
    and the norm_model_y:
    - target_norm_{label}_{T}D.joblib (just applied and saved when norm == True)
    In the file names T represents each time step forecast. 
    returns: (
        df_pred - pandas DataFrame with the predictions, 
        models - dict with the trained models, 
        scx - model used to normalize X_train,  
        scy - dict with the models to normalize the targets)
    '''

    T = horizon_forecast

    if len(X_train) > len(targets[T]):
        X_train_t = X_train.iloc[: len(targets[T])]
        tgt = targets[T]
    else:
        X_train_t = X_train
        tgt = targets[T][:len(X_train_t)]

    if norm:
        scx = norm_model_x
        X_train_t = scx.fit_transform(X_train_t)
        scy = norm_model_y
        tgt = scy.fit_transform(tgt.values.reshape(-1,1))
        model.fit(X_train_t, tgt.ravel())

        if save:
            if path is None:
                dump(model, f'trained_model_norm_{label}_{T}D.joblib')
                dump(scy, f'target_norm_{label}_{T}D.joblib')

            else:
                dump(model, f'{path}trained_model_norm_{label}_{T}D.joblib')
                dump(scy, f'{path}target_norm_{label}_{T}D.joblib') 

    else:
        model.fit(X_train_t, tgt)
        scx = None
        scy = None 
        if save:
            if path is None:
                dump(model, f'trained_model_{label}_{T}D.joblib')

            else:
                dump(model, f'{path}trained_model_{label}_{T}D.joblib')

    if isinstance(X_test, pd.core.frame.DataFrame):

        if norm:
            pred = model.predict(scx.transform(X_test.iloc[: len(targets[T])]))

            pred = np.array(scy.inverse_transform(pred.reshape(-1,1)))[:,0]

        else:
            pred = model.predict(X_test.iloc[: len(targets[T])])


    # transformando preds em um array
    if isinstance(X_test, pd.core.frame.DataFrame):
        train_size = len(X_train)

        y = pred

        # IT'S USED T-1 BECAUSE THE COUNT OF THE INDEX START IN 0, AND NOT 1. 
        x = pd.period_range(
            start=X_test.index[T-1], end=X_test.index[-1], freq="D"
        ).to_timestamp()

        x = np.array(x)

        y = np.array(y)

        target = targets[T]

        dif = len(x) - len(y)

        if dif < 0:
            y = y[: len(y) + dif]

        df_pred = pd.DataFrame()
        df_pred["target"] = target.values
        df_pred["date"] = x
        df_pred["predict"] = y
        df_pred["train_size"] = [train_size] * len(df_pred)
        df_pred.set_index('date', inplace = True)
        df_pred.index = pd.to_datetime(df_pred.index)

    if isinstance(X_test, pd.core.frame.DataFrame) == False:
        df_pred = pd.DataFrame()

    return df_pred, model, scx, scy


def rolling_forecast_one(X_test, horizon_forecast=14, norm = False, path = None, label= 'test'):
    ''''
    Funtion to forecast the data. This function used a pre saved model, for this reason it's necessary 
    to provide the path, label, horizon_forecast and norm configurations used in the function rolling_predictions_mul
    params X_test: Dataframe (with colummns equal in the X_train used in the other function) and just
                   {horizon_forecast} lines
    params horizon_forecast: number of forecast time steps that will be forecast
    params path: folder to load the models 
    params label: label to load the models. 
    returns: dataframe with one column with the values forecasted and with the dates associated
    with the forecast in the index. 
    '''
    forecast_dates = []

    last_day = datetime.strftime((X_test.index)[-1], "%Y-%m-%d")

    a = datetime.strptime(last_day, "%Y-%m-%d")

    for i in np.arange(1, horizon_forecast + 1):

        d_i = datetime.strftime(a + timedelta(days=int(i)), "%Y-%m-%d")

        forecast_dates.append(datetime.strptime(d_i, "%Y-%m-%d"))

    T = horizon_forecast 

    if norm:
        if path is None:
            scx = load(f'feature_norm_{label}.joblib')
            model = load(f'trained_model_norm_{label}_{T}D.joblib')
            scy = load(f'target_norm_{label}_{T}D.joblib')
        else:
            scx = load(f'{path}feature_norm_{label}.joblib')
            model = load(f'{path}trained_model_norm_{label}_{T}D.joblib')
            scy = load(f'{path}target_norm_{label}_{T}D.joblib')

        pred = model.predict(scx.transform(X_test))
        pred = np.array(scy.inverse_transform(pred.reshape(-1,1)))[:,0]
               
    else:
        if path is None:
            model = load(f'trained_model_{label}_{T}D.joblib')
        else:
            model = load(f'{path}trained_model_{label}_{T}D.joblib')
        
        pred = model.predict(X_test)


    df_for = pd.DataFrame()
    df_for["date"] = forecast_dates
    df_for["forecast"] = np.array(pred)
    df_for.set_index('date', inplace = True)
    df_for.index = pd.to_datetime(df_for.index)
    

    return df_for 


def train_eval_mult_models(
    model,
    target_name,
    data,
    ini_date="2020-03-01",
    split=0.75,
    horizon_forecast=14,
    maxlag=14,
    norm_model_x =  MinMaxScaler(), norm_model_y =  MinMaxScaler(),  norm = False,
    save = False, path = None, label= 'test'):

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
        #maxlag=maxlag,
        norm_model_x =  norm_model_x,
        norm_model_y =  norm_model_y,
        norm = norm,
        save = save,
        path = path,
        label = label
    )[0]

    return df_pred


def train_mult_models(
    model,
    target_name,
    data,
    ini_date="2020-03-01",
    horizon_forecast=14,
    maxlag=14,
    norm_model_x =  MinMaxScaler(), norm_model_y =  MinMaxScaler(),  norm = False,
    save = False, path = None, label= 'test'):

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


    idx_preds = pd.period_range(
        start=df_lag.index[0], end=df_lag.index[-1], freq=f"{T}D"
    ).to_timestamp()

    df_pred = rolling_predictions_mult(
        model,
        idx_preds,
        df_lag,
        None,
        targets,
        horizon_forecast=horizon_forecast,
        #maxlag=maxlag,
        norm_model_x =  norm_model_x,
        norm_model_y =  norm_model_y,
        norm = norm,
        save = save,
        path = path,
        label = label
    )[0]

    return df_pred


def forecast_mult_models(
    data,
    horizon_forecast=14,
    maxlag=21, norm = False,
    path = None, label= 'test'):
    ''''
    Funtion to forecast the data. This function used pre saved models for this reason it's necessary 
    to provide the path, label, horizon_forecast and norm configurations used in the function
    train_mult_models
    params data: Dataframe 
    params horizon_forecast: number of forecast time steps that will be forecast
    params max_lag: number of past information that will be used in the forecast
    params path: folder to load the models 
    params label: label to load the models
    returns: dataframe with one column with the values forecasted and with the dates associated
    with the forecast in the index. 
    The difference between this function and the function rolling_forecast_mult is that this function
    transform the input dataframe in the lagged dataframe, which is accepted by the ML models. 
    '''

    df_lag = build_lagged_features(copy.deepcopy(data), maxlag=maxlag)

    df_lag = df_lag.dropna()

    # remove the target column and columns related with the day that we want to predict
    df_lag = df_lag.drop(data.columns, axis=1)

    df_for = rolling_forecast_mult(df_lag.iloc[-1:], 
    horizon_forecast= horizon_forecast, 
    norm = norm, 
    path = path, 
    label= label)

    return df_for 


def train_eval_one_model(
    model,
    target_name,
    data,
    ini_date="2020-01-01",
    split=0.75,
    horizon_forecast=14,
    maxlag=14,
    norm_model_x =  MinMaxScaler(), norm_model_y =  MinMaxScaler(),  norm = False,
    save = False, path = None,  label= 'test'):

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

    idx_preds = pd.period_range(
        start=df_lag.index[0], end=df_lag.index[-1], freq=f"{T}D"
    ).to_timestamp()

    df_pred = rolling_predictions_one(
        model,
        idx_preds,
        X_train,
        df_lag,
        targets,
        horizon_forecast = horizon_forecast,
        #maxlag=maxlag,
        norm_model_x = norm_model_x,
        norm_model_y = norm_model_y,
        norm = norm,
        save = save, 
        path = path,
        label = label 
    )[0]

    return df_pred

def train_one_model(
    model,
    target_name,
    data,
    ini_date="2020-01-01",
    horizon_forecast=14,
    maxlag=14,
    norm_model_x =  MinMaxScaler(), norm_model_y =  MinMaxScaler(),  norm = False,
    save = False, path = None,  label= 'test'):

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


    idx_preds = pd.period_range(
        start=df_lag.index[0], end=df_lag.index[-1], freq=f"{T}D"
    ).to_timestamp()

    df_pred = rolling_predictions_one(
        model,
        idx_preds,
        df_lag,
        None,
        targets,
        horizon_forecast = horizon_forecast,
        #maxlag=maxlag,
        norm_model_x = norm_model_x,
        norm_model_y = norm_model_y,
        norm = norm,
        save = save, 
        path = path,
        label = label 
    )[0]

    return df_pred

def forecast_one_model(
    data,
    horizon_forecast=14,
    maxlag=21, norm = False,
    path = None, label= 'test'):
    ''''
    Funtion to forecast the data. This function used a pre saved model for this reason it's necessary 
    to provide the path, label, horizon_forecast and norm configurations used in the function
    train_one_model
    params data: Dataframe 
    params horizon_forecast: number of forecast time steps that will be forecast
    params max_lag: number of past information that will be used in the forecast
    params path: folder to load the models 
    params label: label to load the models
    returns: dataframe with one column with the values forecasted and with the dates associated
    with the forecast in the index. 
    The difference between this function and the function rolling_forecast_one is that this function
    transform the input dataframe in the lagged dataframe, which is accepted by the ML models. 
    '''

    df_lag = build_lagged_features(copy.deepcopy(data), maxlag=maxlag)

    df_lag = df_lag.dropna()

    # remove the target column and columns related with the day that we want to predict
    df_lag = df_lag.drop(data.columns, axis=1)

    df_for = rolling_forecast_one(df_lag.iloc[-horizon_forecast:], 
    horizon_forecast= horizon_forecast, 
    norm = norm, 
    path = path, 
    label= label)

    return df_for
