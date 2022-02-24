#!/usr/bin/env python3
"""
Created on Mon Feb  7 09:32:55 2022

@author: eduardoaraujo
"""
import numpy as np
import pandas as pd
import plotly
import pytest
import matplotlib 
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


def test_compute_clusters(get_df_cases):
    df = get_df_cases
    df.index = pd.to_datetime(df["datum"])

    inc_canton, clusters, all_regions, fig = clustering.compute_clusters(
        df,
        ["geoRegion", "entries"],
        t=0.5,
        drop_georegions=None,
        smooth=True,
        ini_date="2020-03-01",
        plot=True,
    )

    assert len(inc_canton.columns) == len(df["geoRegion"].unique())
    assert inc_canton.empty == False
    assert len(clusters) >= 1
    assert len(all_regions) == len(df["geoRegion"].unique())
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_curves(get_df_cases):
    df = get_df_cases
    df.index = pd.to_datetime(df["datum"])

    inc_canton = df.pivot(columns="geoRegion", values="entries")

    for i in ["CH", "CHFL", "FL"]:
        del inc_canton[i]

    figs = clustering.plot_clusters("cases", inc_canton, [["GE", "FR", "JU"]])

    for i in figs:
        assert isinstance(i, plotly.graph_objs._figure.Figure)
