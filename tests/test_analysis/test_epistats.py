"""
Tests for epistats module
"""
import pytest
import scipy.stats as st
import numpy as np
from numpy.testing import assert_equal

from epigraphhub.analysis.epistats import posterior_prevalence, incidence_rate


@pytest.mark.parametrize(("a", "b"), [(1, 1), (2, 2)])
def test_prevalence(a, b):
    p = posterior_prevalence(1000, 300, a, b)
    assert isinstance(p, st._distn_infrastructure.rv_frozen)
    assert p.mean() == pytest.approx(0.3, 0.1)
    assert p.std() > 0


@pytest.mark.parametrize(("pop", "cases", "expected"),
                         [(1000, 5, 500),
                          ([1000, 5000, 10000], [5, 5, 5], np.array([500, 100, 50]))]
                         )
def test_incidence_rate(pop, cases, expected):
    ir = incidence_rate(pop, cases)
    assert isinstance(ir, np.ndarray)
    assert_equal(ir, expected)
