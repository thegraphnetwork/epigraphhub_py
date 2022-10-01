from typing import Union

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats._result_classes import RelativeRiskResult
from scipy.stats.contingency import relative_risk


def posterior_prevalence(
    pop_size: int, positives: int, a: float = 1, b: float = 1
) -> st.rv_continuous:
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

    Args:
        pop_size: population size
        positives: number of positives
        a: prior beta parameter alpha
        b: prior beta parameter beta

    Returns:
        object: Returns a scipy stats frozen beta distribution that represents the posterior probability of the prevalence
    """
    a, b = 1, 1  # prior beta parameters
    pa = a + positives
    pb = b + pop_size - positives
    return st.beta(pa, pb)


@np.vectorize
def incidence_rate(
    pop_size: int, new_cases: int, scaling: float = 1e5
) -> Union[float, np.ndarray, np.ndarray]:
    """
    incidence is defined as the number of new cases in a population over a period of time, typically 1 year. The incidence rate is also usually scale to 100k people to facilitate comparisons between localities with different populations.
    Parameters
    ----------
    pop_size: population pop_size
    new_cases: number of new cases observed in the period
    scaling: number to scale the rate to. If ommitted, the rate is return as cases per 100k.

    Returns
    -------
    A float or a np.ndarray of floats

    Examples
    --------
    >>> incidence_rate(1000, 5)
    500.0
    >>> incidence_rate([1000,5000,10000], [5,5,5])
    array([500, 100, 50])
    """
    IR = new_cases / pop_size * scaling
    return IR


def risk_ratio(
    exposed_cases: int, exposed_total: int, control_cases: int, control_total: int
) -> RelativeRiskResult:
    """
    Also known as relative risk, computed the risk of contracting a disease given exposure to a risk factor.
    Parameters:
        exposed_cases: number of cases in the exposed group
        exposed_total: size of the exposed group
        control_cases: number of cases in the control group
        control_total: size of the control group
    Returns:
    RelativeRiskResult object

    Examples:
    >>> rr = risk_ratio(27, 122, 44, 487)
    >>> rr.relative_risk
    2.4495156482861398
    >>> rr.confidence_interval(confidence_level=0.95)
    ConfidenceInterval(low=1.5836990926700116, high=3.7886786315466354)
    """
    rr = relative_risk(exposed_cases, exposed_total, control_cases, control_total)
    return rr
