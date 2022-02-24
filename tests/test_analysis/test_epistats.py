"""
Tests for epistats module
"""
import pytest
import scipy.stats as st

from epigraphhub.analysis.epistats import prevalence


@pytest.mark.parametrize(("a", "b"), [(1, 1), (2, 2)])
def test_prevalence(a, b):
    p = prevalence(1000, 300, a, b)
    assert isinstance(p, st._distn_infrastructure.rv_frozen)
    assert p.mean() == pytest.approx(0.3, 0.1)
    assert p.std() > 0
