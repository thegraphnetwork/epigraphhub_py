"""
This module contains functions to plot the result of the models
"""

import matplotlib.pyplot as plt


def plot_val_ngb(
    df, title, xlabel="Date", ylabel="Incidence", path=None, name=None, save=False
):

    """
    Function to plot the output of the model trained in the test and sample data.

    :params df: dataframe with columns target, median, lower, upper and train_size.
    :params title: string. It will be used as title of the plot.
    :params xlabel: string. It will be used as label of the axis x.
    :params ylabel: string. It will be used as label of the axis y.
    :params path: sting. It indicates where save the plot.
    :params name: string. It indicates which name use to save the plot.
    :params save: boolean. If True the plot is saved.

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
        plt.savefig(f"{path}/{name}.png", bbox_inches="tight")

    plt.show()

    return
