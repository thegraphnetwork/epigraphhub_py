"""
The functions in this module allow the user to compute the hierarchical
clusterization between time series curves of a data frame.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.cluster.hierarchy as hcluster
from scipy.signal import correlate, correlation_lags


def get_lag(x, y, maxlags=5, smooth=True):
    """
    Compute the lag and correlation between two series x and y.
    :params x: first curve.
    :params y: second curve.
    :params maxlags: int. Max lag allowed when computing the lag between the curves.
    :params smooth: boolean. Indicates if a moving average of 7 days will be applied or
                    not.
    :returns:
        lag: int. Represent the lag computed between the curves. ]
        corr: float. Represent the correlation computed between the curves.
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


def lag_ccf(a, maxlags=30, smooth=True):
    """
    Calculate the full correlation matrix based on the maximum correlation lag
    :params a: np.array. Matrix to compute the lags and correlations.
    :params maxlags: int. Max lag allowed when computing the lag between the curves.
    :params smooth: boolean. Indicates if a moving average of 7 days will be applied in
                            the data or not.
    :returns:
        cmat: np.array. Matrix with the correlation computed.
        lags: np.array. Matrix with the lags computed.
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


def compute_clusters(
    df,
    columns,
    t,
    drop_georegions=None,
    smooth=True,
    ini_date=None,
    plot=False,
):
    """
    Function to apply a hierarquial clusterization in a dataframe.

    :params df: Dataframe with date time index.

    :param columns: list of strings. The list should have 2 columns. The first need to
                                    refer to a column with different regions associated
                                    with the second column, which represents the curves
                                    we want to compute the correlation.

    :param t: float. Represent the value used to compute the distance between the clusters
                    and so decide the number of clusters returned.

    :param drop_georegions: list. Param with the georegions that wiil be ignored in the
                                clusterization.

    :param smooth: Boolean. If true a rooling average of seven days will be applied to
                    the data.

    :param ini_date: Represent the initial date to start to compute the correlation
                    between the series.

    :param plot: Boolean. If true a dendogram of the clusterization will be returned.

    :return:
        inc_canton: It's a data frame with datetime index where each collumn represent
                    the same timse series curve for different regions.
        cluster: array. It's the array with the computed clusters
        all_regions: array. It'is the array with all the regions used in the
                            clusterization
        fig : matplotlib.Figure. Plot with the dendorgram of the clusterization.
    """

    df.sort_index(inplace=True)

    inc_canton = df.pivot(columns=columns[0], values=columns[1])

    if smooth:
        inc_canton = inc_canton.rolling(7).mean().dropna()

    if ini_date:
        inc_canton = inc_canton.loc[ini_date:]

    if drop_georegions != None:

        for i in drop_georegions:
            del inc_canton[i]

    inc_canton = inc_canton.dropna()

    cm, lm = lag_ccf(inc_canton.values)

    # Plotting the dendrogram
    linkage = hcluster.linkage(cm, method="complete")

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=300)
        hcluster.dendrogram(
            linkage, labels=inc_canton.columns, color_threshold=0.3, ax=ax
        )
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

    grouped = pd.DataFrame(list(zip(ind, inc_canton.columns))).groupby(0)

    clusters = [group[1][1].values for group in grouped]

    all_regions = df[columns[0]].unique()

    return inc_canton, clusters, all_regions, fig


def plot_clusters(
    curve, inc_canton, clusters, ini_date=None, normalize=False, smooth=True
):
    """
    This function plot the curves of the clusters computed in the function
    compute_clusters

    :param curve: string. Name of the curve used to compute the clusters. It Will
                          be used in the title of the plot.

    :param inc_canton: data frame (table) where each column is the name of the
                        georegion and your values is the time series of the curve selected.
                        This param is the first return of the function compute_clusters.

    :param cluster: list or array of the georegions that will want to see in the
                    same plot.

    :param ini_date: string. Filter the interval that the times series start to be plotted.

    :param normalize: Boolean. Decides when normalize the times serie by your biggest
                      value or not.

    :param smooth: Boolean. If True, a rolling average of seven days will be applied
                   in the data.

    :returns: matplotlib figure.
    """

    if smooth:
        inc_canton = inc_canton.rolling(7).mean().dropna()

    if normalize:

        for i in inc_canton.columns:

            inc_canton[i] = inc_canton[i] / max(inc_canton[i])

    figs = []
    for i in clusters:

        fig = px.line(inc_canton[i], render_mode="SVG")
        fig.update_layout(
            xaxis_title="Time (days)", yaxis_title=f"{curve}", title=f"{curve} series"
        )

        fig.show()

        figs.append(fig)

    return figs
