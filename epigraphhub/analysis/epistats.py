from typing import Union

import arviz as az
import numpy as np
import pandas as pd
import plotly.express as px
import pymc3 as pm
import scipy.stats as st


def prevalence(pop_size: int, positives: int, a: float = 1, b: float = 1) -> object:
    """
    Returns the Bayesian posterior prevalence of a disease for a point in time.
    It assumes number of cases follow a binomial distribution with probability described as a beta(a,b) distribution
    Parameters
    ----------
    pop_size : int
        population size
    positives : int
        number of positives
    a : float, optional
        prior beta parameter alpha, by default 1
    b : float, optional
        prior beta parameter beta, by default 1

    Returns
    -------
    object
        Returns a scipy stats frozen beta distribution that represents the posterior probability of the prevalence
    """
    a, b = 1, 1  # prior beta parameters
    pa = a + positives
    pb = b + pop_size - positives
    return st.beta(pa, pb)


def inf_pos_prob_cases_hosp(
    df: pd.DataFrame,
    alpha: float = 0.5,
    beta: float = 0.5,
    draws: int = 2000,
    tune: int = 500,
) -> az.data.inference_data.InferenceData:
    """
    This function compute the posterior probability distribution for the prevalence of cases and the probability of hospitalization over time.

    Parameters
    ----------
    df : pd.DataFrame
        It takes as input a dataframe with four columns:
            - cases: Number of cases over time.
            - hospiotalizations: Number of hospitalizations over time.
            - tests: Number of tests over time.
            - tests_pos: Proportion of the positive tests over time.
        This data frame must have a datetime index.
    alpha:float
        The alpha parameter of the Beta distribution
    beta:float
        The beta parameter of the Beta distribution
    draws: int
        The number of samples to draw.
    tune: int
        Number of iterations to tune. Samplers adjust the step sizes,
        scalings or similar during tuning. Tuning samples will be drawn in addition to the number specified in the
        draws argument, and will be discarded.

    Returns
    -------
    az.data.inference_data.InferenceData
        An array with the posterior probabilities infered.
    """

    with pm.Model() as var_bin:
        prev = pm.Beta("prevalence", alpha, beta, shape=len(df["cases"]))

        cases = pm.Binomial("cases", n=df["tests"].values, p=prev, observed=df["cases"])

        probs = pm.Beta("phosp", alpha, beta, shape=len(df["cases"]))

        hosp = pm.Binomial(
            "hospitalizations", n=df["cases"], p=probs, observed=df["hospitalizations"]
        )

    with var_bin:
        tracevb = pm.sample(draws, tune=tune, return_inferencedata=True)

    return tracevb


def plot_pos_prob_prev(
    df: pd.DataFrame,
    tracevb: az.data.inference_data.InferenceData,
    ci: bool = False,
    save: bool = False,
    name: Union[str, None] = None,
    plot: bool = True,
):
    """
    This function plots the posterior probability distribution of the prevalence
    generated with the `inf_pos_prob_cases_hosp()` function.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with four columns:
        - cases: Number of cases over time.
        - hospitalizations: Number of hospitalizations over time.
        - tests: Number of tests over time.
        - tests_pos: Proportion of the positive tests over time.
        This data frame must have a datetime index.
    tracevb : az.data.inference_data.InferenceData
        the return of the inf_pos_prob_cases_hosp() function.
    ci : bool, optional
        If True the confidence interval is computed, by default False
    save : bool, optional
        If True the plot is saved, by default False

    Returns
    -------
    fig
        A plotly figure.
    """

    Prev_post = pd.DataFrame(
        index=df["cases"].index,
        data={
            "median": tracevb.posterior.prevalence.median(axis=(0, 1)),
            "lower": np.percentile(tracevb.posterior.prevalence, 2.5, axis=(0, 1)),
            "upper": np.percentile(tracevb.posterior.prevalence, 97.5, axis=(0, 1)),
        },
    )

    fig = px.line(Prev_post.rolling(7).mean().dropna())

    if ci:
        fig.add_scatter(
            x=Prev_post.index,
            y=Prev_post.lower,
            mode="none",
            fill="tonexty",
            name="95% CI",
        )

    fig.add_scatter(
        x=df["tests_pos"].index,
        y=df["tests_pos"].values,
        name="Test positivity",
        mode="markers",
    )
    fig.update_layout(
        title="Estimated prevalence of COVID",
        yaxis_title="Prevalence of infected",
        xaxis_title="Time (days)",
    )
    if save:
        if name == None:
            fig.write_image("prevalence_est.png", scale=3)
        else:
            fig.write_image(f"{name}.png", scale=3)

    if plot:
        fig.show()

    return fig


def plot_pos_prob_hosp(
    df: pd.DataFrame,
    tracevb: az.data.inference_data.InferenceData,
    save: bool = False,
    name: Union[str, None] = None,
    plot: bool = False,
):
    """
    This function plots the posterior probability distribution of hospitalization
    generated with the `inf_pos_prob_cases_hosp()` function.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with four columns:
        - cases: Number of cases over time.
        - hospitalizations: Number of hospitalizations over time.
        - tests: Number of tests over time.
        - tests_pos: Proportion of the positive tests over time.
        This data frame must have a datetime index.
    tracevb : az.data.inference_data.InferenceData
        the return of the inf_pos_prob_cases_hosp() function.
    ci : bool, optional
        If True the confidence interval is computed, by default False
    save : bool, optional
        If True the plot is saved, by default False

    Returns
    -------
    fig
        A plotly figure.
    """

    Phosp_post = pd.DataFrame(
        index=df.index,
        data={
            "median": tracevb.posterior.phosp.median(axis=(0, 1)),
            "lower": np.percentile(tracevb.posterior.phosp, 2.5, axis=(0, 1)),
            "upper": np.percentile(tracevb.posterior.phosp, 97.5, axis=(0, 1)),
        },
    )

    fig = px.line(Phosp_post.rolling(7).mean().dropna())

    fig.update_layout(
        title="Estimated Probability of Hospitalization",
        yaxis_title="probability",
    )

    if plot:
        fig.show()

    if save:
        if name == None:
            fig.write_image("prob_hosp_est.png", scale=3)
        else:
            fig.write_image(f"{name}.png", scale=3)

    return fig
