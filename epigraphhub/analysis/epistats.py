from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import scipy.stats as st


def posterior_prevalence(pop_size: int, positives: int, a: float = 1, b: float = 1) -> st.rv_continuous:
    """
    Returns the Bayesian posterior prevalence of a disease for a point in time.
    It assumes number of cases follow a binomial distribution with probability described as a beta(a,b) distribution

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
def incidence_rate(pop_size: int, new_cases: int, scaling: float = 1e5) -> Union[float, np.ndarray, np.ndarray]:
    """
    incidence is defined as the number of new cases in a population over a period of time, typically 1 year. The incidence rate is also usually scale to 100k people to facilitate comparisons between localities withe different populations.
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
