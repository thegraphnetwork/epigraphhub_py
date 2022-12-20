#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 17:47:12 2022

@author: nr
"""

import pytest
from pytest import approx
import numpy as np
import pandas as pd
from epigraphhub.analysis.py_epitools import (
    nan_arr,
    array_as_integer,
    to_1darray,
    fisher_or_pval,
    ageadjust_direct,
    ageadjust_indirect,
    binom_general,
    binom_exact,
    binom_wilson,
    binom_approx,
    kapmeier
    )

def test_nan_arr(): 
    n = 3  # arbitrary size value
    eq = np.isnan(nan_arr(n))
    assert eq.all()

vals_list = [1,2,3]
@pytest.mark.parametrize('arr, ref', [(np.array(vals_list, dtype='float32'), np.array(vals_list, dtype='int')),
                                      (np.array(vals_list, dtype='float64'), np.array(vals_list, dtype='int')),
                                      (np.array(vals_list, dtype='int32'), np.array(vals_list, dtype='int')),
                                      (np.array(vals_list, dtype='int64'), np.array(vals_list, dtype='int'))])
def test_array_as_integer_working_cases(arr, ref):
    res = array_as_integer(arr)
    assert (res == ref).all()
    assert res.dtype in ['int32', 'int64']

@pytest.mark.parametrize('arr', [np.array([0.1, 0.0, 0.0]), 
                                 np.array(["a", 0, 1]),
                                 np.array(["a", 0, 1], dtype='object'),
                                 np.array([True, False, True]),
                                 np.array([0, 1, True], dtype='object')])
def test_array_as_integer_failing_cases(arr):
    with pytest.raises(ValueError):
        array_as_integer(arr)
        
vals_arr = np.array([1,2,3])
@pytest.mark.parametrize('x, ref', [(vals_list, vals_arr), 
                                    (tuple(vals_list), vals_arr),
                                    (np.array(vals_list), vals_arr),
                                    (pd.Series(vals_list),vals_arr), 
                                    (pd.DataFrame(vals_list),vals_arr)])
def test_to_1darray(x, ref):
    assert (to_1darray(x) == ref).all()
    
def test_fisher_or_pval_raise():
    with pytest.raises(ValueError):
        fisher_or_pval(np.array([[1,9],[11,3]]), or_=0.5, alternative='l')
        
@pytest.mark.parametrize('x, or_, alternative, ref', [(np.array([[1,9],[11,3]]), 1, 'less', 0.0013797280926100418),
                                                      (np.array([[1,9],[11,3]]), 0.5, 'less', 0.01572098756226001),
                                                      (np.array([[1,9],[11,3]]), 0, 'less', 1),
                                                      (np.array([[1,9],[11,3]]), float('inf'), 'less', 0),
                                                      (np.array([[1,9],[11,3]]), 1, 'greater', 0.9999663480953022),
                                                      (np.array([[1,9],[11,3]]), 0.1, 'greater', 0.9252908371124625),
                                                      (np.array([[1,9],[11,3]]), 0, 'greater', 0),
                                                      (np.array([[1,9],[11,3]]), float('inf'), 'greater', 1),
                                                      (np.array([[1,9],[11,3]]), 0, 'two-sided', 0),
                                                      (np.array([[1,9],[11,3]]), 1, 'two-sided', 0.0027594561852200875),
                                                      (np.array([[1,9],[11,3]]), 0.5, 'two-sided', 0.017227728418185317),
                                                      (np.array([[1,9],[11,3]]), 0, 'two-sided', 0)]
                         )
def test_fisher_or_pval(x, or_, alternative, ref):
    assert fisher_or_pval(x, or_=or_, alternative=alternative) == approx(ref, abs=1e-17)
    
def test_age_adjust_direct():
    count = np.array([107., 141.,  60.,  40.,  39.,  25.])
    pop = np.array([230061., 329449., 114920.,  39487.,  14208.,   3052.])
    stdpop = np.array([ 63986.6, 186263.6, 157302.2,  97647. ,  47572.6,  12262.6])
    res = ageadjust_direct(count, pop, stdpop)
    refvals = np.array([0.000563475054603742, 0.000923045255331994, 0.000804417498507042,
           0.001057633457492612])
    assert res.index.tolist() == ['crude_rate', 'adj_rate', 'lci', 'uci']
    assert np.allclose(res.values, refvals, atol=1e-16)

def test_age_adjust_indirect():
    count = np.array([   45.,   201.,   320.,   670.,  1126.,  3160.,  9723., 17935.,
           22179., 13461.,  2238.])
    pop = np.array([  906897.,  3794573., 10003544., 10629526.,  9465330.,  8249558.,
            7294330.,  5022499.,  2920220.,  1019504.,   142532.])
    stdcount = np.array([  141.,   926.,  1253.,  1080.,  1869.,  4891., 14956., 30888.,
       41725., 26501.,  5928.])
    stdpop = np.array([ 1784033.,  7065148., 15658730., 10482916.,  9939972., 10563872.,
            9114202.,  6850263.,  4702482.,  1874619.,   330915.])
    sir_series, rate_series = ageadjust_indirect(count, pop, stdcount, stdpop)
    assert sir_series.index.tolist() == ['observed', 'exp', 'sir', 'lci', 'uci']
    sir_vals = np.array([7.105800000000000e+04, 8.555688884452569e+04,
           8.305351089744144e-01, 8.244509020475589e-01,
           8.366642155718712e-01])
    assert np.allclose(sir_series.values, sir_vals, atol=1e-16, rtol=1e-3)  
    # can be done with np.testing.assert_allclose but I think this is more immediate and easy to notice (e.g. assert is highlighted)
    assert rate_series.index.tolist() == ['crude_rate', 'adj_rate', 'lci', 'uci']
    rate_vals = np.array([0.001195286415322112, 0.001379414537278219, 0.001369309433481852,
           0.001389594213790028])
    assert np.allclose(rate_series.values, rate_vals, atol=1e-16, rtol=1e-3)  

refvals_exact = np.array([[1.0000000000000000e+00, 1.0000000000000000e+01,
        1.0000000000000001e-01, 2.5285785444625728e-03,
        4.4501611702819543e-01, 9.4999999999999996e-01],
       [2.0000000000000000e+00, 2.0000000000000000e+01,
        1.0000000000000001e-01, 1.2348527170295887e-02,
        3.1698271401907918e-01, 9.4999999999999996e-01],
       [3.0000000000000000e+00, 3.0000000000000000e+01,
        1.0000000000000001e-01, 2.1117137029722604e-02,
        2.6528845047420818e-01, 9.4999999999999996e-01],
       [4.0000000000000000e+00, 4.0000000000000000e+01,
        1.0000000000000001e-01, 2.7925415294219370e-02,
        2.3663739987594820e-01, 9.4999999999999996e-01],
       [5.0000000000000000e+00, 5.0000000000000000e+01,
        1.0000000000000001e-01, 3.3275093589021511e-02,
        2.1813536643420220e-01, 9.4999999999999996e-01],
       [6.0000000000000000e+00, 6.0000000000000000e+01,
        1.0000000000000001e-01, 3.7591269108324499e-02,
        2.0505773670843330e-01, 9.4999999999999996e-01],
       [7.0000000000000000e+00, 7.0000000000000000e+01,
        1.0000000000000001e-01, 4.1159701583584785e-02,
        1.9524565254167767e-01, 9.4999999999999996e-01],
       [8.0000000000000000e+00, 8.0000000000000000e+01,
        1.0000000000000001e-01, 4.4170940154062761e-02,
        1.8756510746333893e-01, 9.4999999999999996e-01],
       [9.0000000000000000e+00, 9.0000000000000000e+01,
        1.0000000000000001e-01, 4.6755315274164336e-02,
        1.8135997288393194e-01, 9.4999999999999996e-01],
       [1.0000000000000000e+01, 1.0000000000000000e+02,
        1.0000000000000001e-01, 4.9004689221485952e-02,
        1.7622259774017532e-01, 9.4999999999999996e-01]])
refvals_wilson = np.array([[1.0000000000000000e+00, 1.0000000000000000e+01,
        1.0000000000000001e-01, 1.7876213095072896e-02,
        4.0415002679523848e-01, 9.4999999999999996e-01],
       [2.0000000000000000e+00, 2.0000000000000000e+01,
        1.0000000000000001e-01, 2.7866481213768224e-02,
        3.0103364522848725e-01, 9.4999999999999996e-01],
       [3.0000000000000000e+00, 3.0000000000000000e+01,
        1.0000000000000001e-01, 3.4599888747334176e-02,
        2.5621082579184079e-01, 9.4999999999999996e-01],
       [4.0000000000000000e+00, 4.0000000000000000e+01,
        1.0000000000000001e-01, 3.9579528682606363e-02,
        2.3051775227522295e-01, 9.4999999999999996e-01],
       [5.0000000000000000e+00, 5.0000000000000000e+01,
        1.0000000000000001e-01, 4.3475764931890412e-02,
        2.1360231437479654e-01, 9.4999999999999996e-01],
       [6.0000000000000000e+00, 6.0000000000000000e+01,
        1.0000000000000001e-01, 4.6642834473884162e-02,
        2.0149464723978772e-01, 9.4999999999999996e-01],
       [7.0000000000000000e+00, 7.0000000000000000e+01,
        1.0000000000000001e-01, 4.9289301442539526e-02,
        1.9232914848567401e-01, 9.4999999999999996e-01],
       [8.0000000000000000e+00, 8.0000000000000000e+01,
        1.0000000000000001e-01, 5.1547615567380814e-02,
        1.8510688806104086e-01, 9.4999999999999996e-01],
       [9.0000000000000000e+00, 9.0000000000000000e+01,
        1.0000000000000001e-01, 5.3506751697124919e-02,
        1.7924174875433652e-01, 9.4999999999999996e-01],
       [1.0000000000000000e+01, 1.0000000000000000e+02,
        1.0000000000000001e-01, 5.5229137060675081e-02,
        1.7436566150491345e-01, 9.4999999999999996e-01]])
refvals_approx = np.array([[ 1.0000000000000000e+00,  1.0000000000000000e+01,
         1.0000000000000001e-01, -8.5938509691368459e-02,
         2.8593850969136847e-01,  9.4999999999999996e-01],
       [ 2.0000000000000000e+00,  2.0000000000000000e+01,
         1.0000000000000001e-01, -3.1478381086487234e-02,
         2.3147838108648724e-01,  9.4999999999999996e-01],
       [ 3.0000000000000000e+00,  3.0000000000000000e+01,
         1.0000000000000001e-01, -7.3516486230294220e-03,
         2.0735164862302943e-01,  9.4999999999999996e-01],
       [ 4.0000000000000000e+00,  4.0000000000000000e+01,
         1.0000000000000001e-01,  7.0307451543157734e-03,
         1.9296925484568422e-01,  9.4999999999999996e-01],
       [ 5.0000000000000000e+00,  5.0000000000000000e+01,
         1.0000000000000001e-01,  1.6845770539019325e-02,
         1.8315422946098070e-01,  9.4999999999999996e-01],
       [ 6.0000000000000000e+00,  6.0000000000000000e+01,
         1.0000000000000001e-01,  2.4090921287100400e-02,
         1.7590907871289962e-01,  9.4999999999999996e-01],
       [ 7.0000000000000000e+00,  7.0000000000000000e+01,
         1.0000000000000001e-01,  2.9721849172380826e-02,
         1.7027815082761918e-01,  9.4999999999999996e-01],
       [ 8.0000000000000000e+00,  8.0000000000000000e+01,
         1.0000000000000001e-01,  3.4260809456756386e-02,
         1.6573919054324363e-01,  9.4999999999999996e-01],
       [ 9.0000000000000000e+00,  9.0000000000000000e+01,
         1.0000000000000001e-01,  3.8020496769543853e-02,
         1.6197950323045615e-01,  9.4999999999999996e-01],
       [ 1.0000000000000000e+01,  1.0000000000000000e+02,
         1.0000000000000001e-01,  4.1201080463798390e-02,
         1.5879891953620162e-01,  9.4999999999999996e-01]])
x = np.arange(1,11)
n = 10*x
conf_level, p = 0.95, 0.5
@pytest.mark.parametrize("x, n, conf_level, p, method, refvals", 
                         [(x, n, conf_level, p, "exact", refvals_exact),
                          (x, n, conf_level, p, "wilson", refvals_wilson)])
def test_binom_general(x, n, conf_level, p, method, refvals):
    res = binom_general(x, n, conf_level, p, method)
    assert res.columns.tolist() == ['x', 'n', 'proportion', 'lower', 'upper', 'conf_level']
    assert np.allclose(res.values, refvals, atol=1e-17, rtol=1e-3)    

@pytest.mark.parametrize("x, n, conf_level, p, refvals", [(x, n, conf_level, p, refvals_exact)])
def test_binom_exact(x, n, conf_level, p, refvals):
    res = binom_exact(x, n, conf_level, p)
    assert res.columns.tolist() == ['x', 'n', 'proportion', 'lower', 'upper', 'conf_level']
    assert np.allclose(res.values, refvals, atol=1e-17, rtol=1e-3) 

@pytest.mark.parametrize("x, n, conf_level, p, refvals", [(x, n, conf_level, p, refvals_wilson)])
def test_binom_wilson(x, n, conf_level, p, refvals):
    res = binom_wilson(x, n, conf_level, p)
    assert res.columns.tolist() == ['x', 'n', 'proportion', 'lower', 'upper', 'conf_level']
    assert np.allclose(res.values, refvals, atol=1e-17, rtol=1e-3) 
    
@pytest.mark.parametrize("x, n, conf_level, refvals", [(x, n, conf_level, refvals_approx)])                         
def test_binom_approx(x, n, conf_level, refvals):
    res = binom_approx(x, n, conf_level)
    assert res.columns.tolist() == ['x', 'n', 'proportion', 'lower', 'upper', 'conf_level']

def test_kapmeier():
    time = np.array([ 1., 17., 20.,  9., 24., 16.,  2., 13., 10.,  3.])
    status = np.array([1., 1., 1., 1., 0., 0., 0., 1., 0., 1.])
    res = kapmeier(time, status)
    refvals = np.array([[ 1.                 , 10.                 ,  1.                 ,
             0.9                ,  0.9                ,  0.09999999999999998],
           [ 3.                 ,  8.                 ,  1.                 ,
             0.875              ,  0.7875             ,  0.21250000000000002],
           [ 9.                 ,  7.                 ,  1.                 ,
             0.8571428571428571 ,  0.6749999999999999 ,  0.32500000000000007],
           [13.                 ,  5.                 ,  1.                 ,
             0.8                ,  0.5399999999999999 ,  0.4600000000000001 ],
           [17.                 ,  3.                 ,  1.                 ,
             0.6666666666666666 ,  0.35999999999999993,  0.6400000000000001 ],
           [20.                 ,  2.                 ,  1.                 ,
             0.5                ,  0.17999999999999997,  0.8200000000000001 ]])
    assert res.columns.tolist() == ['time', 'n_risk', 'n_events', 'condsurv', 'survival', 'risk']
    assert np.allclose(res.values, refvals, rtol=1e-17, atol=1e-3)