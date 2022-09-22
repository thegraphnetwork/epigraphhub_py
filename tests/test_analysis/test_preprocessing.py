#!/usr/bin/env python3
import datetime

import numpy as np
import pytest

from epigraphhub.analysis import preprocessing


def test_lagged_features(get_df_test):
    lag = 2
    df_test = get_df_test
    df_lag = preprocessing.build_lagged_features(df_test, lag, True)

    assert df_lag.shape[0] >= 0
    assert df_lag.shape[1] == df_test.shape[1] * (lag + 1)


def test_get_targets(get_df_test):
    s = get_df_test["hosp_GE"]
    predict_n = 4

    targets = preprocessing.get_targets(s, predict_n)

    assert isinstance(targets, dict)
    assert len(targets.keys()) == predict_n
    assert list(targets.keys())[-1] == predict_n


def test_get_next_n_days():
    ini_date = "2020-01-01"
    next_days = 3
    dates = preprocessing.get_next_n_days(ini_date, next_days)

    assert len(dates) == next_days
    assert dates[-1] == datetime.datetime(2020, 1, 4, 0, 0)


def test_lstm_split_data(get_df_test):
    df = get_df_test

    look_back = 12
    ratio = 0.8
    predict_n = 5
    Y_column = 0

    X_train, Y_train, X_test, Y_test = preprocessing.lstm_split_data(
        df, look_back, ratio, predict_n, Y_column
    )

    X_shape = X_train.shape
    Y_shape = Y_train.shape

    assert X_shape[1] == look_back
    assert X_shape[2] == df.shape[1]
    assert Y_shape[1] == predict_n
    assert (
        Y_train[0] == np.array(df.iloc[look_back : look_back + predict_n, Y_column])
    ).all()


def test_norm_data(get_df_test):
    df = get_df_test

    df_norm, df_max = preprocessing.normalize_data(df)

    assert max(df_norm.max()) <= 1.0
    assert min(df_norm.min()) >= 0
