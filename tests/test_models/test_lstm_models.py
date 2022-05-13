"""
Created on Tue Feb  8 09:35:48 2022

@author: eduardoaraujo
"""

import numpy as np
import pandas as pd
import pytest
import keras 
from epigraphhub.models import lstm_models

def test_build_model():
    hidden = 8
    model = lstm_models.build_model(hidden, 9, predict_n=14, look_back=21)

    assert isinstance(model, keras.engine.functional.Functional )

def test_training_eval_model(get_df_test):

    df = get_df_test
    target_name = "hosp_GE"

    model = lstm_models.build_model(8, df.shape[1], predict_n=2, look_back=4)

    df_preds = lstm_models.training_eval_model(model, 
                   target_name,
                   df,
                   ini_date = None,
                   split=0.8,
                   predict_n=2,
                   look_back=4,
                   batch  =1, 
                   epochs = 5,
                   path = None, 
                   label = "train_eval_region_name",
                   uncertainty = True,
                   save = False)

    df_preds = df_preds.dropna()

    df_preds.to_csv("tests/data_for_test/df_preds_lstm")

    assert not df_preds.empty
    assert df_preds.shape[1] == 5

def test_training_model(get_df_test):

    df = get_df_test
    target_name = "hosp_GE"

    model = lstm_models.build_model(8, df.shape[1], predict_n=2, look_back=4)

    model, hist = lstm_models.training_model(model, 
                   target_name,
                   df,
                   ini_date = None,
                   predict_n = 2,
                   look_back=4,
                   batch  =1, 
                   epochs = 1,
                   path = "tests/data_for_test/", 
                   label = "train_lstm_test",
                   save = True)

    assert isinstance(model, keras.engine.functional.Functional )
    assert isinstance(hist,keras.callbacks.History )

def test_forecasting_model(get_df_test): 

    df = get_df_test

    target_name = "hosp_GE"

    df_for = lstm_models.forecasting_model(target_name,
                      df,
                      predict_n=2,
                      look_back=4,
                      path ="tests/data_for_test/", 
                      label = "train_lstm_test_1",
                      uncertainty = True)

    assert not df_for.empty
    assert df_for.shape[1] == 3
