"""
The functions in this module allow the user to compute the hierarchical
clusterization between time series curves of a DataFrame.
"""

from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.cluster.hierarchy as hcluster
from scipy.signal import correlate, correlation_lags


def get_lag(
    x: np.array, y: np.array, maxlags: int = 5, smooth: bool = True
) -> Tuple[int, float]:
    """
    Compute the lag and correlation between two series x and y.

    Parameters
    ----------
    x : np.array
        First curve.
    y : np.array
        Second curve.
    maxlags : int, optional
        Max lag allowed when computing the lag between the curves, by default 5.
    smooth : bool, optional
        Indicates if a moving average of 7 days will be applied or not, by default True.

    Returns
    -------
    Tuple[int, float]
        The first parameter returned is the max lag computed and the second the max
        correlation computed between the curves.
    """

    if smooth:
        x = pd.Series(x).rolling(7).mean().dropna().values
        y = pd.Series(y).rolling(7).mean().dropna().values
    corr = correlate(x, y, mode="full") / np.sqrt(np.dot(x, x) * np.dot(y, y))
    slice = np.s_[(len(corr) - maxlags) // 2 : -(len(corr) - maxlags) // 2]
    corr = corr[slice]
    lags = correlation_lags(x.size, y.size, mode="full")
    lags = lags[slice]
    lag = lags[np.argmax(corr)]
    #     lag = np.argmax(corr)-(len(corr)//2)

    return lag, corr.max()


def plot_xcorr(
    inc: pd.DataFrame,
    X: str,
    Y: str,
    ini_date: Union[str, None] = None,
    smooth: bool = True,
    plot: bool = True,
):
    """
    Plots the Cross correlation between two series identifying the lag.

    Parameters
    ----------
    inc : pd.DataFrame
        A DataFrame with datetime index where each column represent a different
        time series.
    X : str
        The name of a column.
    Y : str
        The name of another column.
    ini_date : str
        A date represented as string to initiate the computation of the
        correlation between the series.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        A plotly.express figure.
    """

    if ini_date != None:
        inc = inc.loc[ini_date:]

    if smooth:
        x = inc[X].rolling(7).mean().dropna().values
        y = inc[Y].rolling(7).mean().dropna().values
    else:
        x = inc[X].values
        y = inc[Y].values

    corr = correlate(x, y, mode="full") / np.sqrt(np.dot(x, x) * np.dot(y, y))
    lags = correlation_lags(x.size, y.size, mode="full")
    lag = lags[np.argmax(corr)]  # -((len(corr)//2))
    fig = px.line(
        x=lags, y=corr, render_mode="SVG"  # np.arange(-len(corr)//2,len(corr)//2)
    )

    fig.update_layout(
        xaxis_title="lag(days)",
        yaxis_title="Cross-correlation",
        title=f"Cross-correlation function between cantons {X} and {Y}. Lag: {lag} days.\nSince: {ini_date}",
    )
    fig.add_shape(
        type="line",
        yref="y",
        xref="x",
        x0=lag,
        y0=0,
        x1=lag,
        y1=corr.max() * 1.05,
        line=dict(color="black", width=1),
    )
    fig.add_annotation(
        x=lag,
        y=1.06,
        yref="paper",
        showarrow=False,
        text=f"Max correlation: {corr.max():2f}",
    )

    if plot:
        fig.show()

    return fig


def lag_ccf(
    a: np.array, maxlags: int = 30, smooth: bool = True
) -> Tuple[np.array, np.array]:
    """
    Calculate the full correlation matrix based on the maximum correlation lag.

    Parameters
    ----------
    a : np.array
        Matrix to compute the lags and correlations.
    maxlags : int, optional
        Max lag allowed when computing the lag between the curves., by default 30
    smooth : bool, optional
        Indicates if a moving average of 7 days will be applied in the data or not.
        By default True.

    Returns
    -------
    Tuple[np.array,np.array]
        cmat: np.array
            Matrix with the correlation computed.
        lags: np.array
            Matrix with the lags computed.
    """

    ncols = a.shape[1]
    lags = np.zeros((ncols, ncols))
    cmat = np.zeros((ncols, ncols))
    for i in range(ncols):
        for j in range(ncols):
            lag, corr = get_lag(a.T[i], a.T[j], maxlags, smooth)
            cmat[i, j] = corr
            lags[i, j] = lag
    return cmat, lags


def plot_matrix(
    cmat: np.array, columns: list, title: str, label_scale: str, plot: bool = True
):
    """
    Plot a heatmap using the values in cmat.

    Parameters
    ----------
    cmat : np.array
        A matrix.
    columns : list
        The list with the names to be used in the figure.
    title : str
        The title of the figure.
    label_scale :str
        The name in the color scale bar.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        A plotly figure.

    """
    fig = px.imshow(
        cmat,
        labels=dict(x="Region", y="Region", color=label_scale),
        x=columns,
        y=columns,
        title=title,
    )
    fig.update_layout(width=600, height=600)

    if plot:
        fig.show()

    return fig


def compute_clusters(
    df: pd.DataFrame,
    columns: list,
    t: float,
    drop_values: Union[list, None] = None,
    smooth: bool = True,
    ini_date: Union[str, None] = None,
    plot: bool = False,
) -> Tuple[pd.DataFrame, np.array, np.array, plt.figure]:
    """
    Function to apply a hierarchical clusterization in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index.
    columns : list
        The list should have 2 columns. The first need to refer to a column with
        different regions associated with the second column, which represents
        the curves we want to compute the correlation.
    t : float
        Represent the value used to compute the distance between the clusters
        and so decide the number of clusters returned.
    drop_values : Union[list,None], optional
        Param with the georegions that will be ignored in the clusterization. By
        default None.
    smooth : bool, optional
        If true a rolling average of seven days will be applied to the data. By
        default True.
    ini_date : Union[str, None], optional
        Represent the initial date to start to compute the correlation between
        the series. By default None.
    plot : bool, optional
        If true a dendrogram of the clusterization will be returned. By default
        False.

    Returns
    -------
    Tuple[pd.DataFrame, np.array, np.array, plt.figure]
        inc_canton: pd.DataFrame
            It's a DataFrame with datetime index where each column represent the
            same time series curve for different regions.
        cluster: np.array
            It's the array with the computed clusters.
        all_regions: np.array
            It's the array with all the regions used in the clusterization.
        fig: matplotlib.Figure
            Plot with the dendrogram of the clusterization.
    """

    df.sort_index(inplace=True)

    inc = df.pivot(columns=columns[0], values=columns[1])

    if smooth:
        inc = inc.rolling(7).mean().dropna()

    if ini_date:
        inc = inc.loc[ini_date:]

    if drop_values != None:

        for i in drop_values:
            del inc[i]

    inc = inc.dropna()

    cm = lag_ccf(inc.values)[0]

    # Plotting the dendrogram
    linkage = hcluster.linkage(cm, method="complete")

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=300)
        hcluster.dendrogram(linkage, labels=inc.columns, color_threshold=0.18, ax=ax)
        ax.set_title(
            "Result of the hierarchical clustering of the series",
            fontdict={"fontsize": 20},
        )
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

    else:
        fig = None

    # computing the cluster
    ind = hcluster.fcluster(linkage, t, "distance")

    grouped = pd.DataFrame(list(zip(ind, inc.columns))).groupby(0)

    clusters = [group[1][1].values for group in grouped]

    all_regions = df[columns[0]].unique()

    return inc, clusters, all_regions, fig


def plot_clusters(
    curve: str,
    inc_canton: pd.DataFrame,
    clusters: np.array,
    ini_date: str = None,
    normalize: bool = False,
    smooth: bool = True,
    plot: bool = True,
) -> list:
    """
    This function plot the curves of the clusters computed in the function
    compute_clusters.

    Parameters
    ----------
    curve : str
        Name of the curve used to compute the clusters. It will be used in the
        plot title.
    inc_canton : pd.DataFrame
        DataFrame (table) where each column is the name of the georegion and
        your values are the time series of the curve selected. This parameter is
        the first return value of the compute_clusters function.
    clusters : np.array
        Array of the georegions that will want to see in the same plot.
    ini_date : str, optional
         Filter the interval from which the time series will be plotted. By
         default None.
    normalize : bool, optional
        Decides when normalize the times serie by your biggest value or not. By
        default False.
    smooth : bool, optional
        If True, a rolling average of seven days will be applied in the data. By
        default True.

    Returns
    -------
    list
        List of matplotlib figure.
    """

    if smooth:
        inc_canton = inc_canton.rolling(7).mean().dropna()

    if ini_date != None:
        inc_canton = inc_canton[ini_date:]

    if normalize:

        for i in inc_canton.columns:
            inc_canton[i] = inc_canton[i] / max(inc_canton[i])

    figs = []
    for i in clusters:

        fig = px.line(inc_canton[i], render_mode="SVG")
        fig.update_layout(
            xaxis_title="Time (days)", yaxis_title=f"{curve}", title=f"{curve} series"
        )

        if plot:
            fig.show()

        figs.append(fig)

    return figs
