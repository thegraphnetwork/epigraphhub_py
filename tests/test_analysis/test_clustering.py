#!/usr/bin/env python3

import matplotlib
import numpy as np
import pandas as pd
import plotly
import pytest

from epigraphhub.analysis import clustering


def test_get_lag():
    lag = 2
    x = pd.Series(np.random.normal(size=100))
    y = x.shift(-lag)

    lag_comp, corr_comp = clustering.get_lag(x, y, smooth=False)

    assert abs(lag_comp) == lag
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
        drop_values=None,
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

    figs = clustering.plot_clusters(
        "cases", inc_canton, [["GE", "FR", "JU"]], plot=False
    )

    for i in figs:
        assert isinstance(i, plotly.graph_objs._figure.Figure)


def test_plot_xcorr(get_df_cases):

    df = get_df_cases
    df.index = pd.to_datetime(df["datum"])

    inc_canton = df.pivot(columns="geoRegion", values="entries")

    fig = clustering.plot_xcorr(inc_canton, X="GE", Y="BE", plot=False)

    assert isinstance(fig, plotly.graph_objs._figure.Figure)


def test_plot_matrix(get_df_cases):
    df = get_df_cases
    df.index = pd.to_datetime(df["datum"])

    inc_canton = df.pivot(columns="geoRegion", values="entries")

    for i in ["CH", "CHFL", "FL"]:
        del inc_canton[i]

    cm, lm = clustering.lag_ccf(inc_canton.values)

    fig_cor = clustering.plot_matrix(
        cm, inc_canton.columns, "Correlation", label_scale="Correlation", plot=False
    )

    fig_lag = clustering.plot_matrix(
        lm, inc_canton.columns, "Lag", label_scale="Lag", plot=False
    )

    assert isinstance(fig_cor, plotly.graph_objs._figure.Figure)
    assert isinstance(fig_lag, plotly.graph_objs._figure.Figure)
