#!/usr/bin/env python3
"""
Created on Mon Feb  7 09:32:55 2022

@author: eduardoaraujo
"""
import numpy as np
import pandas as pd
import plotly
import pytest

from epigraphhub.analysis import clustering


def test_get_lag():
    lag = 2
    x = pd.Series(np.random.normal(size=100))
    y = x.shift(-lag)

    lag_comp, corr_comp = clustering.get_lag(x, y)

    assert lag_comp == lag
    assert isinstance(corr_comp, np.float64)


def test_lag_ccf(get_df_cases):
    df = get_df_cases
    df.index = pd.to_datetime(df["datum"])

    inc_canton = df.pivot(columns="geoRegion", values="entries")

    for i in ["CH", "CHFL", "FL"]:
        del inc_canton[i]

    cm, lm = clustering.lag_ccf(inc_canton.values)

    assert lm.shape == (len(inc_canton.columns), len(inc_canton.columns))
    assert cm.shape == (len(inc_canton.columns), len(inc_canton.columns))


def test_plot_curves(get_df_cases):
    df = get_df_cases
    df.index = pd.to_datetime(df["datum"])

    inc_canton = df.pivot(columns="geoRegion", values="entries")

    for i in ["CH", "CHFL", "FL"]:
        del inc_canton[i]

    fig = clustering.plot_clusters("cases", inc_canton, [["GE", "FR", "JU"]])

    assert isinstance(fig, plotly.graph_objs._figure.Figure)
