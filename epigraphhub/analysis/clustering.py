"""
The functions in this module allow the user to compute the hierarchical 
clusterization between time series curves of all the regions of the 
specified country.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.cluster.hierarchy as hcluster
from scipy.signal import correlate, correlation_lags

from epigraphhub.data.get_data import get_georegion_data


def get_lag(x, y, maxlags=5, smooth=True):
    """
    Compute the lag and correlation between two series x and y
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


# @st.cache
def lag_ccf(a, maxlags=30, smooth=True):
    """
    Calculate the full correlation matrix based on the maximum correlation lag
    """
    ncols = a.shape[1]
    lags = np.zeros((ncols, ncols))
    cmat = np.zeros((ncols, ncols))
    for i in range(ncols):
        for j in range(ncols):
            #             if j>i:
            #                 continue
            lag, corr = get_lag(a.T[i], a.T[j], maxlags, smooth)
            cmat[i, j] = corr
            lags[i, j] = lag
    return cmat, lags


def compute_clusters(
    country,
    curve,
    columns,
    t,
    drop_georegions=None,
    smooth=True,
    ini_date=None,
    plot=False,
):
    """
    Function to compute the clusters of the data

    params country: contry that we want to make the cluster.

    param curve: string. Represent the curve that will used to cluster the regions.

    param columns: list with 3 columns names in this order: ['date_colum',
    'georegion_column', 'target_column']

    param t: float. Represent the value used to compute the distance between the cluster and so
    decide the number of clusters returned.

    param drop_georegions: list. Param with the georegions that wiil be ignored in the
    clusterization.

    param smooth: Boolean. If true a rooling average of seven days will be applied to
    the data.

    param ini_date: Represent the initial date to start to compute the correlation
    between the series.

    param plot: Boolean. If true a dendogram of the clusterization will be returned.

    return: array.

    -> cluster: is the array with the computed clusters
    -> all_regions: is the array with all the regions
    """

    country = country.lower()

    df = get_georegion_data(country, "All", curve, columns)
    df.index = pd.to_datetime(df[columns[0]])

    inc_canton = df.pivot(columns=columns[1][1:-1], values=columns[2])

    if smooth:
        inc_canton = inc_canton.rolling(7).mean().dropna()

    if ini_date:
        inc_canton = inc_canton.loc[ini_date:]

    if drop_georegions != None:

        for i in drop_georegions:
            del inc_canton[i]

    # Computing the correlation matrix based on the maximum correlation lag
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

    grouped = pd.DataFrame(list(zip(ind, inc_canton.columns))).groupby(
        0
    )  # preciso entender melhor essa linha do código
    clusters = [group[1][1].values for group in grouped]

    all_regions = df.geoRegion.unique()

    return inc_canton, clusters, all_regions, fig


def plot_clusters(
    curve, inc_canton, clusters, ini_date=None, normalize=False, smooth=True
):
    """
    This function plot the curves of the clusters computed in the function compute_clusters

    :param curve: string with the name of the curve used to compute the clusters. Will
                  be used in the title of the plot
    :param inc_canton: dataframe (table) where each column is the name of the georegion and your
                        values is the time series of the curve selected. This param is
                        the first return of the function compute_clusters.
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

    for i in clusters:

        fig = px.line(inc_canton[i], render_mode="SVG")
        fig.update_layout(
            xaxis_title="Time (days)", yaxis_title=f"{curve}", title=f"{curve} series"
        )

        fig.show()

    return fig


# if __name__ == '__main__':
#    country = 'Switzerland'
#    curve = 'cases'
#    columns = ['datum', '\"geoRegion\"', 'entries']
#    t = 0.8

#    clusters, all_regions, fig = compute_clusters(country, curve, columns, t, plot = False)
