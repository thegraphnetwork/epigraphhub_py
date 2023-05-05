#!/usr/bin/env python3
"""
The functions in this module allow the application of the ngboost
regressor model. There are separate methods to train and evaluate
(separate the data in train and test datasets), train with all the data
available, and make forecasts.
"""

from typing import Union

import copy
import os

import numpy as np
import pandas as pd
from joblib import dump, load
from ngboost import NGBRegressor
from ngboost.distns import LogNormal
from ngboost.learners import default_tree_learner
from ngboost.scores import LogScore
from sklearn.model_selection import train_test_split

from epigraphhub.analysis.preprocessing import (
    get_next_n_days,
    get_targets,
    preprocess_data,
)

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


class NGBModel:
    """
    This class allows the user to create a ngboost model. The methods
    allows to train and evaluate the model, to train and save the model
    and make the forecast using saved models.
    """

    def __init__(
        self,
        look_back: int,
        predict_n: int,
        validation_split: float,
        early_stop: int,
        params_model: dict = params_model,
    ) -> None:
        """
        Parameters
        ----------
        look_back : int
            Number of the last days that will be used to forecast the
            next days.
        predict_n : int
            Number of days that will be predicted.
        validation_split : float
            Proportion of training data to set aside as validation data
            for early stopping.
        early_stop : int
            The number of consecutive boosting iterations during which
            the loss has to increase before the algorithm stops early.
            Set to None to disable early stopping and validation. None
            enables running over the full data set.
        params_model : dict
            The dict with the params that will be used in the ngboost
            regressor model.
        """

        # This will remove the validation_fraction and early_stopping_rounds parameters since it shouldn't.
        # This parameters was included int the .fit() step to allow this class to be use by users with version
        # 0.3.12 (current available in pip) of ngboost.
        for i in ["validation_fraction", "early_stopping_rounds"]:
            if i in params_model.keys():
                del params_model[i]

        self.look_back = look_back
        self.predict_n = predict_n
        self.validation_split = validation_split
        self.early_stop = early_stop
        self.ngb_model = NGBRegressor(**params_model)

    def train_eval(
        self,
        target_name: str,
        data: pd.DataFrame,
        ini_date: Union[str, None] = None,
        end_train_date: Union[str, None] = None,
        end_date: Union[str, None] = None,
        ratio: float = 0.75,
        path: Union[str, None] = None,
        name: str = "train_eval_ngb",
        save: bool = False,
    ) -> pd.DataFrame:
        """
        Function to apply a ngboost regressor model given a dataset and
        a target column. This function will train multiple models, each
        one specialist in predict the X + n days, of the target column,
        where n is in the range (1, number of days that you want
        predict). This function split the data in train and test dataset
        and returns the predictions made using the test dataset.

        Parameters
        ----------
        target_name : str
            Name of the target column.
        data : pd.DataFrame
            DataFrame with features and target column.
        ini_date : str, optional
            Determines the beginning of the train dataset, by default
            None.
        end_train_date : str, optional
            Determines the beginning of end of train dataset. If is not
            None, then ratio isn't used, by default None.
        end_date : str, optional
            Determines the end of the dataset used in validation, by
            default None.
        ratio : float
            Determines which percentage of the data will be used to
            train the model, by default 0.75.
        path : str, optional
            It indicates where save the models trained, by default None.
        name : str, optional
            It indicates which name use to save the models trained, by
            default None.
        save : bool
            If True the models trained are saved, by default False.

        Returns
        -------
        pd.DataFrame
            A DataFrame with four columns (and a date index):

            - target: The target values.
            - lower: The lower value of the confidence interval of 95%.
            - median: The median value of the confidence interval of
              95%.
            - upper: The upper value of the confidence interval of 95%.
            - train_size: The number of rows of data using as training
              data.
        """

        df_lag = preprocess_data(data, self.look_back, ini_date, end_date)

        target = df_lag[target_name]

        targets = get_targets(target, self.predict_n)

        # remove the target column and columns related with the day that we want to predict
        df_lag = df_lag.drop(data.columns, axis=1)

        if end_train_date == None:
            X_train = df_lag.iloc[: int(df_lag.shape[0] * ratio)]

        else:
            X_train = df_lag.loc[:end_train_date]

        idx = pd.period_range(
            start=df_lag.index[0], end=df_lag.index[-1], freq=f"{self.predict_n}D"
        )

        idx = idx.to_timestamp()

        preds5 = np.empty((len(idx), self.predict_n))
        preds50 = np.empty((len(idx), self.predict_n))
        preds95 = np.empty((len(idx), self.predict_n))

        for T in range(1, self.predict_n + 1):

            if len(X_train) <= len(targets[T]):
                tgt = targets[T][: len(X_train)]

            else:
                tgt = targets[T]

            model = copy.deepcopy(self.ngb_model)

            X_train_t, X_val, y_train, y_val = train_test_split(
                X_train[: len(tgt)], tgt, test_size=self.validation_split
            )

            model.fit(
                X_train_t,
                y_train,
                X_val=X_val,
                Y_val=y_val,
                early_stopping_rounds=self.early_stop,
            )

            pred = model.pred_dist(df_lag.loc[idx], max_iter=model.best_val_loss_itr)

            pred50 = pred.median()

            pred5, pred95 = pred.interval(alpha=0.95)

            preds5[:, (T - 1)] = pred5
            preds50[:, (T - 1)] = pred50
            preds95[:, (T - 1)] = pred95

            if save:
                if path != None:
                    dump(model, f"{path}/{name}_{T}.joblib")
                else:
                    dump(model, f"{name}_{T}.joblib")

        train_size = len(X_train) + len(X_val)

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

    def train(
        self,
        target_name: str,
        data: pd.DataFrame,
        ini_date: Union[str, None] = None,
        end_date: Union[str, None] = None,
        save: bool = True,
        path: Union[str, None] = None,
        name: str = "train_ngb",
    ) -> list:
        """
        Function to train multiple ngboost regressor models given a
        dataset and a target column. This function will train multiple
        models, each one specialist in predict the X + n days, of the
        target column, where n is in the range (1, number of days that
        you want predict). This function will train the model with all
        the data available and will save the model that will be used to
        make forecasts.

        Parameters
        ----------
        target_name : str
            Name of the target column.
        data : pd.DataFrame
            DataFrame with features and target column.
        ini_date : str, optional
            Determines the beginning of the train dataset, by default
            None.
        end_date : str, optional
            Determines the end of the train dataset, by default None.
        save : bool
            If True the models is saved, by default True.
        path : str, optional
            Indicates where the models will be saved, by default
            "../opt/models/saved_models/ml".

        Returns
        -------
        list
            A list with the trained models.
        """

        predict_n = self.predict_n

        df_lag = preprocess_data(data, self.look_back, ini_date, end_date)

        target = df_lag[target_name]

        models = []

        # remove the target column and columns related with the day that we want to predict
        df_lag = df_lag.drop(data.columns, axis=1)

        targets = get_targets(target, predict_n)

        X_train = df_lag.iloc[:-1]

        for T in range(1, predict_n + 1):
            tgt = targets[T]

            model = copy.deepcopy(self.ngb_model)

            model.fit(X_train.iloc[: len(tgt)], tgt)

            if save:
                if path != None:
                    dump(model, f"{path}/{name}_{T}.joblib")
                else:
                    dump(model, f"{name}_{T}.joblib")

            models.append(model)

        return models

    def forecast(
        self,
        data: pd.DataFrame,
        end_date: Union[str, None] = None,
        name: str = "train_ngb",
        path: Union[str, None] = None,
    ) -> pd.DataFrame:

        """
        Function to load multiple ngboost regressor model trained with
        the function `training_model` and make the forecast.

        Important: predict_n and max_lag need have the same value used
        in training_model. Only the last that of the dataset will be
        used to forecast the next predict_n days.

        Parameters
        ----------
        target_name : str
            Name of the target column.
        data : pd.DataFrame
            DataFrame with features and target column.
        ini_date : str, optional
            Determines the beginning of the train dataset, by default
            None.
        end_date : str, optional
            Determines the end of the train dataset, by default None.
        path : str, optional
            Indicates where the models will be saved, by default
            "../opt/models/saved_models/ml".

        Returns
        -------
        pd.DataFrame
            A DataFrame with three columns regarding (and a date index):

            - lower: The lower value of the confidence interval of 95%.
            - median: The median value of the confidence interval of
              95%.
            - upper: The upper value of the confidence interval of 95%.
        """

        df_lag = preprocess_data(data, self.look_back, None, end_date)

        # remove the target column and columns related with the day that we want to predict
        df_lag = df_lag.drop(data.columns, axis=1)

        forecasts5 = []
        forecasts50 = []
        forecasts95 = []

        for T in range(1, self.predict_n + 1):

            if path == None:
                filename = f"{name}_{T}.joblib"
            else:
                filename = f"{path}/{name}_{T}.joblib"

            if os.path.exists(filename):

                model = load(filename)

                forecast = model.pred_dist(df_lag.iloc[-1:])

                forecast50 = forecast.median()

                forecast5, forecast95 = forecast.interval(alpha=0.95)

                forecasts5.append(forecast5)
                forecasts50.append(forecast50)
                forecasts95.append(forecast95)

            else:
                raise Exception("The saved models was not found.")

        forecast_dates = get_next_n_days(str(df_lag.index[-1])[:10], self.predict_n)

        df_for = pd.DataFrame()
        df_for["date"] = forecast_dates
        df_for["lower"] = np.array(forecasts5)
        df_for["median"] = np.array(forecasts50)
        df_for["upper"] = np.array(forecasts95)
        df_for.set_index("date", inplace=True)

        return df_for
