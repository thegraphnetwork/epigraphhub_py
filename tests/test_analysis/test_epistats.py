"""
Tests for epistats module
"""
import pytest
import scipy.stats as st
import numpy as np
from numpy.testing import assert_equal

from epigraphhub.analysis.epistats import posterior_prevalence, incidence_rate, risk_ratio


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


def test_relative_risk():
    result = risk_ratio(27, 122, 44, 487)
    ci = result.confidence_interval(confidence_level=0.95)
    assert result.relative_risk == 2.4495156482861398
    assert ci.low == 1.5836990926700116
    assert ci.high == 3.7886786315466354
