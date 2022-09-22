#!/usr/bin/env python3
"""
Created on Mon Feb  7 08:31:21 2022

@author: eduardoaraujo
"""

import pandas as pd
import pytest


@pytest.fixture
def get_df_test():
    df_test = pd.read_csv("tests/data_for_test/data_test.csv")
    df_test.set_index("datum", inplace=True)
    df_test.index = pd.to_datetime(df_test.index)
    return df_test


@pytest.fixture
def get_df_cases():
    df = pd.read_csv("tests/data_for_test/df_cases.csv")
    return df


@pytest.fixture
def get_df_inf():
    df = pd.read_csv("tests/data_for_test/data_to_inf.csv")
    df.set_index("datum", inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.loc["2021-01-01":"2022-01-01"]
    return df
