"""
This module contains functions to plot the result of the forecast models.
"""
import matplotlib.pyplot as plt
import pandas as pd


def plot_val(
    df: pd.DataFrame,
    title: str,
    xlabel: str = "Date",
    ylabel: str = "Incidence",
    path: str = None,
    name: str = None,
    save: bool = False,
) -> None:
    """
    Function to plot the output of the model trained in the test and sample
    data.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with a datetime index and four columns: 'target', 'lower',
        'median', 'upper' and 'train_size'.
    title : str
        Title of the plot.
    xlabel : str, optional
        Title of the axis x, by default "Date".
    ylabel : str, optional
        Title of the axis y, by default "Incidence".
    path : str, optional
        Folder to save the plot, by default None.
    name : str, optional
        Name used to save the plot, by default None.
    save : bool, optional
        If True the plot is saved, by default False.
    """

    fig, ax = plt.subplots(dpi=150)

    ax.plot(df.target, label="Data", color="black")
    ax.plot(df["median"], label="Predict", color="tab:orange")
    ax.fill_between(df.index, df.lower, df.upper, color="tab:orange", alpha=0.5),

    ax.axvline(
        df.index[df.train_size[0]],
        0,
        max(df[["target", "lower", "median", "upper"]].max()),
        label="Train/Test",
        color="green",
        ls="--",
    )

    ax.legend()
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for label in ax.get_xticklabels():
        label.set_rotation(30)

    if save:
        if path == None:
            plt.savefig(f"{name}.png", bbox_inches="tight")
        else:
            plt.savefig(f"{path}/{name}.png", bbox_inches="tight")

    plt.show()

    return


def plot_forecast(
    curve: pd.DataFrame,
    df_for: pd.DataFrame,
    title: str,
    xlabel: str = "Date",
    ylabel: str = "Incidence",
    path: str = None,
    name: str = None,
    save: bool = False,
) -> None:
    """
    Function to plot the output of the model trained in the test and sample
    data.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with a datetime index and four columns: 'target', 'lower',
        'median', 'upper' and 'train_size'.
    title : str
        Title of the plot.
    xlabel : str, optional
        Title of the axis x, by default "Date".
    ylabel : str, optional
        Title of the axis y, by default "Incidence".
    path : str, optional
        Folder to save the plot, by default None.
    name : str, optional
        Name used to save the plot, by default None.
    save : bool, optional
        If True the plot is saved, by default False.
    """

    plt.figure()

    plt.plot(curve, label="Data", color="black")

    plt.plot(df_for["median"], label="Median", color="tab:orange")

    plt.fill_between(
        df_for.index, df_for.upper, df_for.lower, color="tab:orange", alpha=0.5
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.title(title)
    plt.xticks(rotation=25)
    plt.legend()
    plt.grid()

    if save:
        if path == None:
            plt.savefig(f"{name}.png", bbox_inches="tight")
        else:
            plt.savefig(f"{path}/{name}.png", bbox_inches="tight")

    plt.show()

    return
