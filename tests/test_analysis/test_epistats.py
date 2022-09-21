"""
Tests for epistats module
"""

import arviz as az
import plotly
import pytest
import scipy.stats as st

from epigraphhub.analysis.epistats import (
    inf_pos_prob_cases_hosp,
    plot_pos_prob_hosp,
    plot_pos_prob_prev,
    prevalence,
)


@pytest.mark.parametrize(("a", "b"), [(1, 1), (2, 2)])
def test_prevalence(a, b):
    p = prevalence(1000, 300, a, b)
    assert isinstance(p, st._distn_infrastructure.rv_frozen)
    assert p.mean() == pytest.approx(0.3, 0.1)
    assert p.std() > 0


def test_inf_pos_prob_cases_hosp(get_df_inf):

    inf = inf_pos_prob_cases_hosp(get_df_inf, draws=100, tune=10)

    assert isinstance(inf, az.data.inference_data.InferenceData)


def test_plots_pos(get_df_inf):

    inf = inf_pos_prob_cases_hosp(get_df_inf, draws=100, tune=10)

    fig1 = plot_pos_prob_prev(get_df_inf, inf, plot=False)
    fig2 = plot_pos_prob_hosp(get_df_inf, inf, plot=False)

    assert isinstance(fig1, plotly.graph_objs._figure.Figure)
    assert isinstance(fig2, plotly.graph_objs._figure.Figure)
