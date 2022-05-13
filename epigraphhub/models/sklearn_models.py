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
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from epigraphhub.data.preprocessing import build_lagged_features

def rolling_predictions_one(
    model, target_name, data, ini_date = '2020-03-01', split = 0.75, horizon_forecast=14,
    maxlag = 14, norm_model_x =  MinMaxScaler(), norm_model_y =  MinMaxScaler(),  norm = False,
    save = False, path = None, label= 'test'):
    '''
    This function train one model trained to predict the X + {horizon_forecast} day. 
     When we train the model with all the data, that's X_test = None the function return 
     a empty dataframe. 

    :param model:  model to apply in the data. It should be passed a model compatible with the sklearn methods. 
    :param target_name:string. Name of the target column.
    :param data: dataframe. Dataframe with features and target column.
    :param ini_date: string. Determines the beggining of the train dataset
    :param split: float. Determines which percentage of the data will be used to train the model
    :param horizon_forecast: int. Number of days that will be predicted
    :param max_lag: int. Number of the last days that will be used to forecast the next days
    :param norm: boolean. If true the data is normalized before training
    :param norm_model_x: model to normalize the features (X_train and X_test datasets)
    :param norm_model_y: model to normalize the targets (arrays in targets dict)
    :params save:boolean. Indicates if the models will be saved in some path or not
    :params path: string. Indicates the path where the models will be saved
    :params label: possibilite add a different name annotation for the  models saved. 


    returns: 
    :return df_pred: pandas DataFrame with the target values and the predictions
    :return models: trained model
    :return X_train: array with features used training the model
    :return targets: dictionary with the target values param norm: decide if the data will be normalized 

    By default the model is saved as:
    * `trained_model_norm_{label}_{horizon_forecast}D.joblib` (when norm == True);
    * `trained_model_{label}_{horizon_forecast}D.jobli` (when norm == False).

    If `norm = True`, it's also necessary to save the models used to normalize the features and targets. In this case, these models are saved as: 
    * `features_norm_{label}.joblib`, for the `norm_model_x` (just applied and saved when norm == True);
    * `target_norm_{label}D.joblib`, for the `norm_model_y` (just applied and saved when norm == True).

    and the returns:
    * `df_pred`: DataFrame with the target values and the predictions. The data frame has the following columns: `target`, with the values used to train and test the model, 	`predict`, with the predicted values and `train_size`, with the number of target observations used to train the model. The dataframe also have a datetime index. Type: `pandas Dataframe`;
    * `model`: the trained and tested model;
    * `scx`: If norm = True, this represents the model used to normalize the features. Otherwise the return is None;
    * `scy`: If norm = True, this represents the model used to normalize the targets. Otherwise the return is None. 
    '''

    T = horizon_forecast

    target = data[target_name]

    df_lag = build_lagged_features(copy.deepcopy(data), maxlag=maxlag)

    ini_date = max(
        df_lag.index[0], target.index[0], datetime.strptime(ini_date, "%Y-%m-%d")
    )

    df_lag = df_lag[ini_date:]
    target = target[ini_date:]
    target = target.dropna()
    df_lag = df_lag.dropna()

    targets = {}

    targets[T] = target.shift(-(T))[:-(T)]

    X_train, X_test, y_train, y_test = train_test_split(
        df_lag, target, train_size=split, test_size=1 - split, shuffle=False
    )

    if len(X_train) > len(targets[T]):
        X_train_t = X_train.iloc[: len(targets[T])]
        tgt = targets[T]
    else:
        X_train_t = X_train
        tgt = targets[T][:len(X_train_t)]

    if norm:
        scx = norm_model_x

        X_train_t = pd.DataFrame(scx.fit_transform(X_train_t), 
                                index = X_train_t.index, columns = X_train_t.columns)

        scy = norm_model_y
        tgt = scy.fit_transform(tgt.values.reshape(-1,1))

        model.fit(X_train_t, tgt.ravel())

        if save:
            if path is None:
                dump(model, f'trained_model_norm_{label}_{T}D.joblib')
                dump(scx, f'features_norm_{label}.joblib')
                dump(scy, f'target_norm_{label}_{T}D.joblib')

            else:
                dump(model, f'{path}/trained_model_norm_{label}_{T}D.joblib')
                dump(scx, f'{path}/features_norm_{label}.joblib')
                dump(scy, f'{path}/target_norm_{label}_{T}D.joblib') 

    else:
        model.fit(X_train_t, tgt)
        scx = None
        scy = None 
        if save:
            if path is None:
                dump(model, f'trained_model_{label}_{T}D.joblib')

            else:
                dump(model, f'{path}/trained_model_{label}_{T}D.joblib')

    if isinstance(X_test, pd.core.frame.DataFrame):

        if norm:

            pred = model.predict( pd.concat( [ X_train_t, pd.DataFrame( scx.transform(X_test), index = X_test.index, 
                        columns = X_test.columns  )] ).iloc[: len(targets[T])])
            
            pred = np.array(scy.inverse_transform(pred.reshape(-1,1)))[:,0]

        else:
            pred = model.predict( pd.concat( [X_train_t, X_test] ).iloc[: len(targets[T])])


    if isinstance(X_test, pd.core.frame.DataFrame):
        train_size = len(X_train)

        y = pred

        x = pd.period_range(
            start=df_lag.index[T], end=df_lag.index[-1], freq="D"
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

def rolling_predictions_mult(
    model, target_name,
    data, ini_date="2020-03-01",
    split=0.75, horizon_forecast=14, maxlag=14,
    norm = False, norm_model_x =  MinMaxScaler(),
    norm_model_y =  MinMaxScaler(), 
    save = False, path = None, label= 'test'):
    '''
    This function train `horizon_forecast` models. Each model is trained to predict the value in the X + {T} day, 
    being T in the range \[1, horizon_forecast\]. 

    :param model:  model to apply in the data. It should be passed a model compatible with the sklearn methods. 
    :param target_name:string. Name of the target column.
    :param data: dataframe. Dataframe with features and target column.
    :param ini_date: string. Determines the beggining of the train dataset
    :param split: float. Determines which percentage of the data will be used to train the model
    :param horizon_forecast: int. Number of days that will be predicted
    :param max_lag: int. Number of the last days that will be used to forecast the next days
    :param norm: boolean. If true the data is normalized before training
    :param norm_model_x: model to normalize the features (X_train and X_test datasets)
    :param norm_model_y: model to normalize the targets (arrays in targets dict)
    :params save:boolean. Indicates if the models will be saved in some path or not
    :params path: string. Indicates the path where the models will be saved
    :params label: possibilite add a different name annotation for the  models saved. 


    By default the models trained are saved as:
* 'trained_model_norm_{label}_{T}D.joblib' (when norm == True)
* 'trained_model_{label}_{T}D.joblib' (when norm == False)

for each T in the range \[1, horizon_forecast\]. 

If `norm = True`, it's also necessary to save the models used to normalize the features and targets. In this case, this model are saved as: 
* 'feature_norm_{label}.joblib', for the `norm_model_x` (just applied and saved when norm == True)
* 'target_norm_{label}_{T}D.joblib', for the `norm_model_y` (just applied and saved when norm == True)
for each T in the range \[1, horizon_forecast\]. 

and the returns:
* df_pred. DataFrame with the target values and the predictions. The data frame has the following columns: `target`, with the values used to train and test the model, 	`predict`, with the predicted values and `train_size`, with the number of target observations used to train the model. The dataframe also has a datetime index. Type: `pandas Dataframe`.
* model: dictionary with the models trained. The key value is the number of days ahead that the model was trained to predict. Type:`dictionary`. 
* scx: If norm = True, this represents the model used to normalize the features. Otherwise, the return is None. 
* dict_scy: If norm = True, this represents the dictionary with the models used to normalize the targets. Otherwise, the return is an empty dictionary. 
 
'''


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

    idx_preds = pd.period_range(
            start=ini_date, end=df_lag.index[-1], freq=f"{horizon_forecast}D"
        )

    idx_preds = idx_preds.to_timestamp()

    if isinstance(X_test, pd.core.frame.DataFrame):
        preds = np.empty((len(idx_preds), horizon_forecast))

    if norm: 
        scx = norm_model_x
        X_train = pd.DataFrame(scx.fit_transform(X_train), index = X_train.index, columns = X_train.columns)

        if save:
            if path is None:
                dump(scx, f'features_norm_{label}.joblib')
            else:
                dump(scx, f'{path}/features_norm_{label}.joblib')
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

            dict_scy[T] = scy

            if save: 
                if path is None:
                    dump(model, f'trained_model_norm_{label}_{T}D.joblib')
                    dump(scy, f'target_norm_{label}_{T}D.joblib')

                else:
                    dump(model, f'{path}/trained_model_norm_{label}_{T}D.joblib')
                    dump(scy, f'{path}/target_norm_{label}_{T}D.joblib') 

        else:
            model.fit(X_train_t, tgt)
            if path is None:
                dump(model, f'trained_model_{label}_{T}D.joblib')
            else:
                dump(model, f'{path}/trained_model_{label}_{T}D.joblib')


        models[T] = model

        if  isinstance(X_test, pd.core.frame.DataFrame):

            if norm:
                pred = model.predict( pd.concat( [ X_train_t, pd.DataFrame( scx.transform(X_test), index = X_test.index, 
                        columns = X_test.columns  )] ).loc[idx_preds])
                pred = np.array(scy.inverse_transform(pred.reshape(-1,1)))[:,0]

            else:
                pred = model.predict( pd.concat([X_train_t, X_test]).loc[idx_preds])

            preds[:, (T - 1)] = pred

    # transformando preds em um array
    if isinstance(X_test, pd.core.frame.DataFrame):
        train_size = len(X_train)

        y = preds.flatten()

        x = pd.period_range(
            start= df_lag.index[1], end=df_lag.index[-1], freq="D"
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

def plot_predicted_vs_data(df_pred, title = 'Ngboost predictions', save = False, filename = 'ngboost_pred', path = None):

    '''
    Function to plot the data vs the predictions 
    
    :params df_pred: pandas Dataframe with the following columns: target, lower, median, upper and train_size 
                    (this column is optional since if you train the model with all the data available plot this
                    column is nonsense)
    :params title: string. This string is used as title of the plot
    :params save: booelan. If true the plot is saved
    :params filename: string. Name of the png file where the plot is saved 
    :params path: string|None. Path where the figure must be saved. If None the 
                                figure is saved in the current directory.

    :returns: None  
    '''
    fig, ax = plt.subplots()

    ax.plot(df_pred.target, label = 'Data')
    
    ax.plot(df_pred['predict'], label = 'Predicted', color = 'tab:orange')

    
    if ('train_size' in df_pred.columns):
    
        ax.axvline(df_pred.index[df_pred.train_size[0]], min( df_pred.target.min(), df_pred.predict.min()), 
              
              max( df_pred.target.max(), df_pred.predict.max()), color = 'tab:green', ls = '--', label = 'Train/Test')
    
    ax.set_title(f'{title}')
    
    ax.set_xlabel('Date')
    
    #ax.set_ylabel('Predictions')
    
    for label in ax.get_xticklabels():
        label.set_rotation(30)
    
    ax.legend()
    
    ax.grid()
    
    if save:
        if path == None:
            plt.savefig(f'{filename}.png', dpi = 300, bbox_inches = 'tight')
            
        else:
            plt.savefig(f'{path}/{filename}.png', dpi = 300, bbox_inches = 'tight')
            
    plt.show()
    
    return 

def training_model_one(
    model,
    target_name,
    data,
    ini_date="2020-01-01",
    horizon_forecast=14,
    maxlag=14,
    norm = False,
    norm_model_x =  MinMaxScaler(), norm_model_y =  MinMaxScaler(),
    save = False, path = None,  label= 'test'):

    '''
    This function train one model trained to predict the X + {horizon_forecast} day. 
     When we train the model with all the data, that's X_test = None the function return 
     a empty dataframe. 

    :param model:  model to apply in the data. It should be passed a model compatible with the sklearn methods. 
    :param target_name:string. Name of the target column.
    :param data: dataframe. Dataframe with features and target column.
    :param ini_date: string. Determines the beggining of the train dataset
    :param split: float. Determines which percentage of the data will be used to train the model
    :param horizon_forecast: int. Number of days that will be predicted
    :param max_lag: int. Number of the last days that will be used to forecast the next days
    :param norm: boolean. If true the data is normalized before training
    :param norm_model_x: model to normalize the features (X_train and X_test datasets)
    :param norm_model_y: model to normalize the targets (arrays in targets dict)
    :params save:boolean. Indicates if the models will be saved in some path or not
    :params path: string. Indicates the path where the models will be saved
    :params label: possibilite add a different name annotation for the  models saved. 


    returns: 
    :return df_pred: pandas DataFrame with the target values and the predictions
    :return models: trained model
    :return X_train: array with features used training the model
    :return targets: dictionary with the target values param norm: decide if the data will be normalized 

    By default the model is saved as:
    * `trained_model_norm_{label}_{horizon_forecast}D.joblib` (when norm == True);
    * `trained_model_{label}_{horizon_forecast}D.jobli` (when norm == False).

    If `norm = True`, it's also necessary to save the models used to normalize the features and targets. In this case, these models are saved as: 
    * `features_norm_{label}.joblib`, for the `norm_model_x` (just applied and saved when norm == True);
    * `target_norm_{label}D.joblib`, for the `norm_model_y` (just applied and saved when norm == True).

    and the returns:
    * `model`: the trained and tested model;
    * `scx`: If norm = True, this represents the model used to normalize the features. Otherwise the return is None;
    * `scy`: If norm = True, this represents the model used to normalize the targets. Otherwise the return is None. 
    '''

    target = data[target_name]

    df_lag = build_lagged_features(copy.deepcopy(data), maxlag=maxlag)

    ini_date = max( df_lag.index[0], target.index[0], datetime.strptime(ini_date, "%Y-%m-%d"))

    df_lag = df_lag[ini_date:]
    target = target[ini_date:]
    target = target.dropna()
    df_lag = df_lag.dropna()

    targets = {}

    T = horizon_forecast 

    targets[T] = target.shift(-(T))[: -(T)]
    
    T = horizon_forecast

    if len(df_lag) > len(targets[T]):
        X_train_t = df_lag.iloc[: len(targets[T])]
        tgt = targets[T]
    else:
        X_train_t = df_lag
        tgt = targets[T][:len(X_train_t)]

    if norm:
        scx = norm_model_x

        X_train_t = pd.DataFrame(scx.fit_transform(X_train_t), 
                                index = X_train_t.index, columns = X_train_t.columns)

        scy = norm_model_y
        tgt = scy.fit_transform(tgt.values.reshape(-1,1))

        model.fit(X_train_t, tgt.ravel())

        if save:
            if path is None:
                dump(model, f'trained_model_norm_{label}_{T}D.joblib')
                dump(scx, f'features_norm_{label}.joblib')
                dump(scy, f'target_norm_{label}_{T}D.joblib')

            else:
                dump(model, f'{path}/trained_model_norm_{label}_{T}D.joblib')
                dump(scx, f'{path}/features_norm_{label}.joblib')
                dump(scy, f'{path}/target_norm_{label}_{T}D.joblib') 

    else:
        model.fit(X_train_t, tgt)
        scx = None
        scy = None 
        if save:
            if path is None:
                dump(model, f'trained_model_{label}_{T}D.joblib')

            else:
                dump(model, f'{path}/trained_model_{label}_{T}D.joblib')

    return model, scx, scy


def training_model_mult(
    model,
    target_name,
    data,
    ini_date="2020-01-01",
    horizon_forecast=14,
    maxlag=14,
    norm_model_x =  MinMaxScaler(), norm_model_y =  MinMaxScaler(),  norm = False,
    save = False, path = None,  label= 'test'):

    '''
    This function train `horizon_forecast` models. Each model is trained to predict the value in the X + {T} day, 
    being T in the range \[1, horizon_forecast\]. 

    :param model:  model to apply in the data. It should be passed a model compatible with the sklearn methods. 
    :param target_name:string. Name of the target column.
    :param data: dataframe. Dataframe with features and target column.
    :param ini_date: string. Determines the beggining of the train dataset
    :param split: float. Determines which percentage of the data will be used to train the model
    :param horizon_forecast: int. Number of days that will be predicted
    :param max_lag: int. Number of the last days that will be used to forecast the next days
    :param norm: boolean. If true the data is normalized before training
    :param norm_model_x: model to normalize the features (X_train and X_test datasets)
    :param norm_model_y: model to normalize the targets (arrays in targets dict)
    :params save:boolean. Indicates if the models will be saved in some path or not
    :params path: string. Indicates the path where the models will be saved
    :params label: possibilite add a different name annotation for the  models saved. 


    By default the models trained are saved as:
    * 'trained_model_norm_{label}_{T}D.joblib' (when norm == True)
    * 'trained_model_{label}_{T}D.joblib' (when norm == False)

    for each T in the range \[1, horizon_forecast\]. 

    If `norm = True`, it's also necessary to save the models used to normalize the features and targets. In this case, this model are saved as: 
    * 'feature_norm_{label}.joblib', for the `norm_model_x` (just applied and saved when norm == True)
    * 'target_norm_{label}_{T}D.joblib', for the `norm_model_y` (just applied and saved when norm == True)
    for each T in the range \[1, horizon_forecast\]. 

    and the returns:
    * df_pred. DataFrame with the target values and the predictions. The data frame has the following columns: `target`, with the values used to train and test the model, 	`predict`, with the predicted values and `train_size`, with the number of target observations used to train the model. The dataframe also has a datetime index. Type: `pandas Dataframe`.
    * model: dictionary with the models trained. The key value is the number of days ahead that the model was trained to predict. Type:`dictionary`. 
    * scx: If norm = True, this represents the model used to normalize the features. Otherwise, the return is None. 
    * dict_scy: If norm = True, this represents the dictionary with the models used to normalize the targets. Otherwise, the return is an empty dictionary. 
    
    '''

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

    idx_preds = pd.period_range(
            start=ini_date, end=df_lag.index[-1], freq=f"{horizon_forecast}D"
        )

    idx_preds = idx_preds.to_timestamp()

    if norm: 
        scx = norm_model_x
        X_train = pd.DataFrame(scx.fit_transform(df_lag), index = df_lag.index, columns = df_lag.columns)

        if save:
            if path is None:
                dump(scx, f'feature_norm_{label}.joblib')
            else:
                dump(scx, f'{path}/feature_norm_{label}.joblib')
    else: 
        X_train = df_lag 
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

            dict_scy[T] = scy

            if save: 
                if path is None:
                    dump(model, f'trained_model_norm_{label}_{T}D.joblib')
                    dump(scy, f'target_norm_{label}_{T}D.joblib')

                else:
                    dump(model, f'{path}/trained_model_norm_{label}_{T}D.joblib')
                    dump(scy, f'{path}/target_norm_{label}_{T}D.joblib') 

        else:
            model.fit(X_train_t, tgt)
            if path is None:
                dump(model, f'trained_model_{label}_{T}D.joblib')
            else:
                dump(model, f'{path}/trained_model_{label}_{T}D.joblib')


        models[T] = model

    return models, scx, dict_scy


def rolling_forecast_one(target_name,data, horizon_forecast = 14, maxlag = 14, norm = False, path = None, label= 'test'):
    """
    This function forecasts a time series using, like features, the last observation of the `data` param and the **saved** models trained with `training_model_one()`.
    The `horizon_forecast` and `maxlag` parameters should be the same used in the `training_model_one()`

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
        df_lag.index[0], target.index[0])

    df_lag = df_lag[ini_date:]
    target = target[ini_date:]
    target = target.dropna()
    df_lag = df_lag.dropna()

    forecast_dates = []

    last_day = datetime.strftime((df_lag.index)[-1], "%Y-%m-%d")

    a = datetime.strptime(last_day, "%Y-%m-%d")

    for i in np.arange(1, horizon_forecast + 1):

        d_i = datetime.strftime(a + timedelta(days=int(i)), "%Y-%m-%d")

        forecast_dates.append(datetime.strptime(d_i, "%Y-%m-%d"))

    T = horizon_forecast 

    if norm:
        if path is None:
            scx = load(f'features_norm_{label}.joblib')
            model = load(f'trained_model_norm_{label}_{T}D.joblib')
            scy = load(f'target_norm_{label}_{T}D.joblib')
        else:
            scx = load(f'{path}/features_norm_{label}.joblib')
            model = load(f'{path}/trained_model_norm_{label}_{T}D.joblib')
            scy = load(f'{path}/target_norm_{label}_{T}D.joblib')

        pred = model.predict(scx.transform(df_lag.iloc[-horizon_forecast:]))
        pred = np.array(scy.inverse_transform(pred.reshape(-1,1)))[:,0]
               
    else:
        if path is None:
            model = load(f'trained_model_{label}_{T}D.joblib')
        else:
            model = load(f'{path}/trained_model_{label}_{T}D.joblib')
        
        pred = model.predict(df_lag.iloc[-horizon_forecast:])


    df_for = pd.DataFrame()
    df_for["date"] = forecast_dates
    df_for["forecast"] = np.array(pred)
    df_for.set_index('date', inplace = True)
    df_for.index = pd.to_datetime(df_for.index)
    

    return df_for 

def rolling_forecast_mult(target_name,data, horizon_forecast = 14, maxlag = 14, norm = False, path = None, label= 'test'):
    """
    This function forecasts a time series using, like features, the last observation of the `data` param and the saved models trained with `training_model_mult()`. 
    The `horizon_forecast` and `maxlag` parameters should be the same used in the `training_model_mult()`. 
    
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
        df_lag.index[0], target.index[0])

    df_lag = df_lag[ini_date:]
    target = target[ini_date:]
    target = target.dropna()
    df_lag = df_lag.dropna()

    forecast_dates = []

    last_day = datetime.strftime((df_lag.index)[-1], "%Y-%m-%d")

    a = datetime.strptime(last_day, "%Y-%m-%d")

    for i in np.arange(1, horizon_forecast + 1):

        d_i = datetime.strftime(a + timedelta(days=int(i)), "%Y-%m-%d")

        forecast_dates.append(datetime.strptime(d_i, "%Y-%m-%d"))

    forecasts = []

    if norm:
        if path is None:
            scx = load(f'feature_norm_{label}.joblib')
        else:
            scx = load(f'{path}/feature_norm_{label}.joblib')
               

    for T in range(1, horizon_forecast + 1):

        if norm:
            if path is None:
                model = load(f'trained_model_norm_{label}_{T}D.joblib')
                scy = load(f'target_norm_{label}_{T}D.joblib')
            else:
                model = load(f'{path}/trained_model_norm_{label}_{T}D.joblib')
                scy = load(f'{path}/target_norm_{label}_{T}D.joblib')

        else:
            if path is None:
                model = load(f'trained_model_{label}_{T}D.joblib')
            else:
                model = load(f'{path}/trained_model_{label}_{T}D.joblib')

        if norm:
            pred = model.predict(scx.transform(df_lag.iloc[-1:]))
            pred = np.array(scy.inverse_transform(pred.reshape(-1,1)))[:,0]
        else:
            pred = model.predict(df_lag.iloc[-1:])

        forecasts.append(pred)

    df_for = pd.DataFrame()
    df_for["date"] = forecast_dates
    df_for["forecast"] = np.array(forecasts)
    df_for.set_index('date', inplace = True)
    df_for.index = pd.to_datetime(df_for.index)

    return df_for 
    
def plot_forecast(df, target_name, df_for, last_values = 90, title = 'Forecast', save = False, filename = 'forecast', path = None):

    '''
    Function to plot the forecast 
    
    :params df: pandas Dataframe with the data used to make the forecast
    :params target_name: string. Name of the target column forecasted
    :params df_for: pandas Dataframe with the forecast values. This dataframe must have the 
                    following columns: lower, median, upper and a datetime index 
                    
    :params last_values: int. Number of last values of the df show in the plot. 
          
    :params title: string. This string is used as title of the plot
    :params save: booelan. If true the plot is saved
    :params filename: string. Name of the png file where the plot is saved 
    :params path: string|None. Path where the figure must be saved. If None the 
                                figure is saved in the current directory.

    :returns: None  
    '''
    fig, ax = plt.subplots()

    ax.plot(df[target_name][-last_values:], label = 'Data')
    
    ax.plot(df_for['forecast'], label = 'Forecast', color = 'tab:orange')
    
    ax.set_title(f'{title}')
    
    ax.set_xlabel('Date')
    
    for label in ax.get_xticklabels():
        label.set_rotation(30)
    
    ax.legend()
    
    ax.grid()
    
    if save:
        if path == None:
            plt.savefig(f'{filename}.png', dpi = 300, bbox_inches = 'tight')
            
        else:
            plt.savefig(f'{path}/{filename}.png', dpi = 300, bbox_inches = 'tight')
            
    plt.show()
    
    return 
