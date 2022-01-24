import numpy as np
import pandas as pd
import scipy.stats as st


def prevalence(pop_size: int, positives: int, a: float = 1, b: float = 1) -> object:
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
