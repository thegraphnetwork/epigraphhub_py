#!/usr/bin/env python3
"""
Created on Tue Feb  8 09:35:48 2022

@author: eduardoaraujo
"""

import numpy as np
import pandas as pd

from epigraphhub.analysis.forecast_models import ngboost_models as ngb
from epigraphhub.analysis.forecast_models.metrics import compute_metrics


def test_rolling_predictions(get_df_test):

    df = get_df_test
    target_name = "hosp_GE"

    ngb_m = ngb.NGBModel(predict_n=2, look_back=3, validation_split=0.1, early_stop=10)

    df_preds = ngb_m.train_eval(
        target_name, df, ini_date="2020-03-01", ratio=0.75, save=False
    )

    df_preds = df_preds.dropna()

    df_m = compute_metrics(df_preds)

    assert not df_preds.empty
    assert df_m.shape[1] == 2
    assert len(df_m.index) == 6


def test_training_model(get_df_test):

    df = get_df_test
    target_name = "hosp_GE"

    ngb_m = ngb.NGBModel(predict_n=2, look_back=3, validation_split=0.1, early_stop=10)

    models = ngb_m.train(
        target_name,
        df,
        path="tests/saved_models_for_test/",
        save=True,
    )

    assert len(models) == 2


def test_rolling_forecast(get_df_test):

    df = get_df_test

    ngb_m = ngb.NGBModel(predict_n=2, look_back=3, validation_split=0.1, early_stop=10)

    df_for = ngb_m.forecast(df, path="tests/saved_models_for_test/")

    assert not df_for.empty
