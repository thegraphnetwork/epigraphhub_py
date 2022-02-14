#!/usr/bin/env python3
"""
Created on Tue Feb  8 09:35:48 2022

@author: eduardoaraujo
"""

import numpy as np
import pandas as pd
import pytest

from epigraphhub.models import ngboost_models


def test_rolling_predictions(get_df_test):

    df = get_df_test
    target_name = "hosp_GE"

    df_preds = ngboost_models.rolling_predictions(
        target_name, df, ini_date="2020-03-01", split=0.75, horizon_forecast=2, maxlag=3
    )

    df_preds = df_preds.dropna()

    assert not df_preds.empty


def test_training_model(get_df_test):

    df = get_df_test
    target_name = "hosp_GE"
    models = ngboost_models.training_model(
        target_name,
        df,
        horizon_forecast=2,
        maxlag=3,
        path="tests/data_for_test/",
        save=True,
    )

    assert len(models) == 2


def test_rolling_forecast(get_df_test):

    df = get_df_test

    target_name = "hosp_GE"

    df_for = ngboost_models.rolling_forecast(
        target_name, df, horizon_forecast=2, maxlag=3, path="tests/data_for_test/"
    )

    assert not df_for.empty
