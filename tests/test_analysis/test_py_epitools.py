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
    kapmeier,
    pois_exact,
    pois_daly,
    pois_byar,
    pois_approx,
    or_midp,
    ormidp_test,
    table_margins,
    tab2by2_test,
    epitable
    )
from typing import Union

def assert_pd(o1: Union[pd.Series, pd.DataFrame], o2: Union[pd.Series, pd.DataFrame], **kwargs):
    """
    Asserts wheter the two objects are close, if values are all numerical or nan, or equal, for object type.

    Parameters
    ----------
    o1 : Union[pd.Series, pd.DataFrame]
        first object to compare
    o2 : Union[pd.Series, pd.DataFrame]
        second object to compare.
    kwargs: **kwargs
    Returns
    -------
    None.

    """
    if o1.equals(o2):
        return None
    assert o1.index.tolist() == o2.index.tolist(), 'Different index!'
    assert o1.index.name == o2.index.name,  'Different index name!'
    if type(o1) == pd.DataFrame:
        assert o1.columns.tolist() == o2.columns.tolist(), 'Different columns'
        assert o1.columns.name == o2.columns.name, 'Different columns name'
    try:
        equal_values = np.allclose(o1.values, o2.values, **kwargs)
    except TypeError:
        raise AssertionError('values datatype is object and they are not (exactly) equal!')
    assert equal_values
    
def assert_dict(d1: dict, d2: dict, atol: float = 1e-17, rtol: float = 1e-06,
                equal_nan: bool = True):
    """
    Asserts whether dictionaries are (approximately) equal.
    It checks for equal number of keys, then compares values.
    Pandas DataFrames and Series are compared with .equals,
    strings with "==", anything else with == pytest.approx.
    
    Parameters
    ----------
    d1 : dict
        first dictionary.
    d2 : TYPE
        second dictionary.
    atol : float, optional
        tolerance for absolute difference. The default is 1e-08.
    rtol : float, optional
        tolerance for relative difference. The default is 1e-05.
    equal_nan : bool, optional
        Whether to use True for Nan == Nan. The default is True.

    Returns
    -------
    None.

    """
    kw_approx = {'abs': atol, 'rel': rtol, 'nan_ok': equal_nan}
    kw_allclose = {'atol': atol, 'rtol': rtol, 'equal_nan': equal_nan}
    assert len(d1.keys()) == len(d2.keys())
    for k,v in d1.items():
        if type(v) in [pd.DataFrame, pd.Series]:
            assert_pd(v, d2[k], **kw_allclose)
        elif type(v) == str:
            assert v == d2[k]
        else:
            assert v == approx(d2[k], **kw_approx)

def test_nan_arr(): 
    n = 3  # arbitrary size value
    eq = np.isnan(nan_arr(n))
    assert eq.all()

@pytest.mark.parametrize('arr', [np.array([0.1, 0.0, 0.0]), 
                                 np.array(["a", 0, 1]),
                                 np.array(["a", 0, 1], dtype='object'),
                                 np.array([True, False, True]),
                                 np.array([0, 1, True], dtype='object')])
def test_array_as_integer_raise(arr):
    with pytest.raises(ValueError):
        array_as_integer(arr)

vals_list = [1,2,3]
@pytest.mark.parametrize('arr, ref', [(np.array(vals_list, dtype='float32'), np.array(vals_list, dtype='int')),
                                      (np.array(vals_list, dtype='float64'), np.array(vals_list, dtype='int')),
                                      (np.array(vals_list, dtype='int32'), np.array(vals_list, dtype='int')),
                                      (np.array(vals_list, dtype='int64'), np.array(vals_list, dtype='int'))])
def test_array_as_integer(arr, ref):
    res = array_as_integer(arr)
    assert (res == ref).all()
    assert res.dtype in ['int32', 'int64']
        
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
    ref = pd.Series([0.000563475054603742, 0.000923045255331994, 0.000804417498507042,
           0.001057633457492612], index=['crude_rate', 'adj_rate', 'lci', 'uci'])
    assert_pd(res, ref)

def test_age_adjust_indirect():
    count = np.array([   45.,   201.,   320.,   670.,  1126.,  3160.,  9723., 17935.,
           22179., 13461.,  2238.])
    pop = np.array([  906897.,  3794573., 10003544., 10629526.,  9465330.,  8249558.,
            7294330.,  5022499.,  2920220.,  1019504.,   142532.])
    stdcount = np.array([  141.,   926.,  1253.,  1080.,  1869.,  4891., 14956., 30888.,
       41725., 26501.,  5928.])
    stdpop = np.array([ 1784033.,  7065148., 15658730., 10482916.,  9939972., 10563872.,
            9114202.,  6850263.,  4702482.,  1874619.,   330915.])
    sir_res, rate_res = ageadjust_indirect(count, pop, stdcount, stdpop)
    sir_ref = pd.Series([7.105800000000000e+04, 8.555688884452569e+04,
           8.305351089744144e-01, 8.244509020475589e-01,
           8.366642155718712e-01], index=['observed', 'exp', 'sir', 'lci', 'uci'])
    assert_pd(sir_res, sir_ref)
    rate_ref = pd.Series([0.001195286415322112, 0.001379414537278219, 0.001369309433481852,
           0.001389594213790028], index=['crude_rate', 'adj_rate', 'lci', 'uci'])
    assert_pd(rate_res, rate_ref)

ref_exact = pd.DataFrame(data=np.array([[1.0000000000000000e+00, 1.0000000000000000e+01,
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
        1.7622259774017532e-01, 9.4999999999999996e-01]]),
                         columns=['x', 'n', 'proportion', 'lower', 'upper', 'conf_level'])
                         
ref_wilson = pd.DataFrame(data=np.array([[1.0000000000000000e+00, 1.0000000000000000e+01,
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
        1.7436566150491345e-01, 9.4999999999999996e-01]]),
                         columns=['x', 'n', 'proportion', 'lower', 'upper', 'conf_level'])
ref_approx = pd.DataFrame(data=np.array([[ 1.0000000000000000e+00,  1.0000000000000000e+01,
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
         1.5879891953620162e-01,  9.4999999999999996e-01]]),
                          columns=['x', 'n', 'proportion', 'lower', 'upper', 'conf_level'])
x = np.arange(1,11)
n = 10*x
conf_level, p = 0.95, 0.5
@pytest.mark.parametrize("x, n, conf_level, p, method, ref", 
                         [(x, n, conf_level, p, "exact", ref_exact),
                          (x, n, conf_level, p, "wilson", ref_wilson)])
def test_binom_general(x, n, conf_level, p, method, ref):
    res = binom_general(x, n, conf_level, p, method)
    assert_pd(res, ref)    

@pytest.mark.parametrize("x, n, conf_level, p, ref", [(x, n, conf_level, p, ref_exact)])
def test_binom_exact(x, n, conf_level, p, ref):
    res = binom_exact(x, n, conf_level, p)
    assert_pd(res, ref)   

@pytest.mark.parametrize("x, n, conf_level, p, ref", [(x, n, conf_level, p, ref_wilson)])
def test_binom_wilson(x, n, conf_level, p, ref):
    res = binom_wilson(x, n, conf_level, p)
    assert_pd(res, ref)   
    
@pytest.mark.parametrize("x, n, conf_level, ref", [(x, n, conf_level, ref_approx)])                         
def test_binom_approx(x, n, conf_level, ref):
    res = binom_approx(x, n, conf_level)
    assert_pd(res, ref)   

def test_kapmeier():
    time = np.array([ 1., 17., 20.,  9., 24., 16.,  2., 13., 10.,  3.])
    status = np.array([1., 1., 1., 1., 0., 0., 0., 1., 0., 1.])
    res = kapmeier(time, status)
    ref = pd.DataFrame(data=np.array([[ 1.                 , 10.                 ,  1.                 ,
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
             0.5                ,  0.17999999999999997,  0.8200000000000001 ]]),
                       columns=['time', 'n_risk', 'n_events', 'condsurv', 'survival', 'risk'])
    assert_pd(res, ref)

def test_pois_exact():  
    # NB. I noticed differences of e-06/e-05 between R and Python. I think they stem from the uncertainty in the root finding algortihm.
    # Are we ok with such accuracy or should we aim for higher accuracy? We might want to add a kwarg for the threshold in the root finding.
    res = pois_exact(np.arange(1,11))
    ref = pd.DataFrame(data=np.array([[ 1.        ,  1.        ,  1.        ,  0.02531781,  5.57164339,
             0.95      ],
           [ 2.        ,  1.        ,  2.        ,  0.24220928,  7.22468767,
             0.95      ],
           [ 3.        ,  1.        ,  3.        ,  0.61867212,  8.76727307,
             0.95      ],
           [ 4.        ,  1.        ,  4.        ,  1.08986537, 10.24158868,
             0.95      ],
           [ 5.        ,  1.        ,  5.        ,  1.62348639, 11.66833208,
             0.95      ],
           [ 6.        ,  1.        ,  6.        ,  2.20189425, 13.05947402,
             0.95      ],
           [ 7.        ,  1.        ,  7.        ,  2.81436305, 14.42267536,
             0.95      ],
           [ 8.        ,  1.        ,  8.        ,  3.45383218, 15.76318922,
             0.95      ],
           [ 9.        ,  1.        ,  9.        ,  4.1153731 , 17.08480345,
             0.95      ],
           [10.        ,  1.        , 10.        ,  4.7953887 , 18.39035604,
             0.95      ]]),
                       columns=['x', 'pt', 'rate', 'lower', 'upper', 'conf_level'])
    assert_pd(res, ref)
    
def test_pois_daly():  
    res = pois_daly(np.arange(1,11))
    ref = pd.DataFrame(data=np.array([[ 1.        ,  1.        ,  1.        ,  0.02531781,  5.57164339,
             0.95      ],
           [ 2.        ,  1.        ,  2.        ,  0.24220928,  7.22468767,
             0.95      ],
           [ 3.        ,  1.        ,  3.        ,  0.61867212,  8.76727307,
             0.95      ],
           [ 4.        ,  1.        ,  4.        ,  1.08986537, 10.24158868,
             0.95      ],
           [ 5.        ,  1.        ,  5.        ,  1.62348639, 11.66833208,
             0.95      ],
           [ 6.        ,  1.        ,  6.        ,  2.20189425, 13.05947402,
             0.95      ],
           [ 7.        ,  1.        ,  7.        ,  2.81436305, 14.42267536,
             0.95      ],
           [ 8.        ,  1.        ,  8.        ,  3.45383218, 15.76318922,
             0.95      ],
           [ 9.        ,  1.        ,  9.        ,  4.1153731 , 17.08480345,
             0.95      ],
           [10.        ,  1.        , 10.        ,  4.7953887 , 18.39035604,
             0.95      ]]),
                       columns=['x', 'pt', 'rate', 'lower', 'upper', 'conf_level'])
    assert_pd(res, ref)
    
def test_pois_byar():  
    res = pois_byar(np.arange(1,11))
    ref = pd.DataFrame(data=np.array([[ 1.        ,  1.        ,  1.        ,  0.09069458,  4.66207302,
             0.95      ],
           [ 2.        ,  1.        ,  2.        ,  0.39884141,  6.41083414,
             0.95      ],
           [ 3.        ,  1.        ,  3.        ,  0.83027534,  8.00366989,
             0.95      ],
           [ 4.        ,  1.        ,  4.        ,  1.33731738,  9.51008003,
             0.95      ],
           [ 5.        ,  1.        ,  5.        ,  1.89638763, 10.95955875,
             0.95      ],
           [ 6.        ,  1.        ,  6.        ,  2.49398174, 12.36787791,
             0.95      ],
           [ 7.        ,  1.        ,  7.        ,  3.1215517 , 13.74464162,
             0.95      ],
           [ 8.        ,  1.        ,  8.        ,  3.77329325, 15.09621249,
             0.95      ],
           [ 9.        ,  1.        ,  9.        ,  4.44505618, 16.42706368,
             0.95      ],
           [10.        ,  1.        , 10.        ,  5.13375332, 17.74048212,
             0.95      ]]), columns=['x', 'pt', 'rate', 'lower', 'upper', 'conf_level'])
    assert_pd(res, ref)
    
def test_pois_approx():  
    res = pois_approx(np.arange(1,11))
    ref = pd.DataFrame(data=np.array([[ 1.        ,  1.        ,  1.        , -0.95996398,  2.95996398,
             0.95      ],
           [ 2.        ,  1.        ,  2.        , -0.77180765,  4.77180765,
             0.95      ],
           [ 3.        ,  1.        ,  3.        , -0.3947572 ,  6.3947572 ,
             0.95      ],
           [ 4.        ,  1.        ,  4.        ,  0.08007203,  7.91992797,
             0.95      ],
           [ 5.        ,  1.        ,  5.        ,  0.6173873 ,  9.3826127 ,
             0.95      ],
           [ 6.        ,  1.        ,  6.        ,  1.19908832, 10.80091168,
             0.95      ],
           [ 7.        ,  1.        ,  7.        ,  1.81442272, 12.18557728,
             0.95      ],
           [ 8.        ,  1.        ,  8.        ,  2.4563847 , 13.5436153 ,
             0.95      ],
           [ 9.        ,  1.        ,  9.        ,  3.12010805, 14.87989195,
             0.95      ],
           [10.        ,  1.        , 10.        ,  3.80204968, 16.19795032,
             0.95      ]]), columns=['x', 'pt', 'rate', 'lower', 'upper', 'conf_level'])
    assert_pd(res, ref)

def test_or_midp_raise():
    with pytest.raises(ValueError):
        or_midp(np.arange(16).reshape(4,4))
        
@pytest.mark.parametrize('x, byrow', [(np.array([[12,2],[7,9]]), True), (np.array([12,2,7,9]), True), (np.array([12,7,2,9]), False)])
def test_or_midp(x, byrow):
    res = or_midp(x, byrow=byrow)
    ref = {'x': np.array([[12,  2],
            [ 7,  9]]),
      'estimate': 6.88070083979995,
     'conf_int': (1.276249402536501, 60.72108876785116),
     'conf_level': 0.95,
     'method': 'median-unbiased estimate & mid-p exact CI'}
    assert_dict(ref, res)

def test_ormidp_test():
    res = ormidp_test(12,2,7,9)
    ref = {'one_sided': 0.011660836248542417, 'two_sided': 0.023321672497084833}
    assert_dict(res, ref)

arr = np.array([[1,3],[2,4]])
df = pd.DataFrame(data=arr, index=['ROW1', 'ROW2'], columns=['COL1', 'COL2'])
vals = np.array([[ 1.,  3.,  4.],
       [ 2.,  4.,  6.],
       [ 3.,  7., 10.]])
ref_arr = pd.DataFrame(data=vals, index=['row0', 'row1', 'Total'], columns=['col0', 'col1', 'Total'])
ref_df = pd.DataFrame(data=vals, index=['ROW1', 'ROW2', 'Total'], columns=['COL1', 'COL2', 'Total'])
@pytest.mark.parametrize('x, ref', [(arr, ref_arr), (df, ref_df)])
def test_table_margins(x, ref):
    res = table_margins(x)
    assert res.equals(ref)

def test_tab2by2_test():
    inp = pd.DataFrame(data=np.array([[2,29],[35,64],[12,6]]),
                       index=['Lowest','Intermediate','highest'],
                       columns=['Case', 'Control'])
    inp.index.name = 'Tap water exposure'
    inp.columns.name = 'Outcome'
    res = tab2by2_test(inp)
    pval_df = pd.DataFrame(data=np.array([[np.nan, np.nan, np.nan],
           [1.01865796e-03, 1.26117842e-03, 1.85757180e-03],
           [1.35795799e-05, 1.31816999e-05, 6.85865882e-06]]),
                           index=['Lowest', 'Intermediate', 'highest'],
                           columns=['midp_exact', 'fisher_exact', 'chi_square'])
    pval_df.index.name = 'Tap water exposure'
    ref = {'x': inp,
     'p_value': pval_df,
     'correction': False}
    assert_dict(res, ref, equal_nan=True)

ref1 = pd.DataFrame(data=np.array([[88,20],[555,347]]),
                    index=['Exposed1', 'Exposed2'],
                    columns=['Disease1', 'Disease2'])
ref1.index.name, ref1.columns.name = 'Predictor', 'Outcome'
s1, s2 = ref1['Disease1'], ref1['Disease2']
s2_2 = ref1.reset_index()['Disease2']
ref2 = pd.DataFrame(data=np.array([[1, 2], [3, 4], [5, 6]]),
                    index=['Exposed1', 'Exposed2', 'Exposed3'],
                    columns=['Disease1', 'Disease2'])
ref3 = pd.DataFrame(data=np.array([[1, 2, 3], [4, 5, 6]]),
                    index=['Exposed1', 'Exposed2'],
                    columns=['Disease1', 'Disease2', 'Disease3'])
v1 = np.array(['H', 'L', 'M', 'M', 'L', 'H', 'H', 'L', 'L'])
v2 = np.array(['Y', 'N', 'Y', 'Y', 'N', 'N', 'Y', 'Y', 'N'])
ref4 = pd.DataFrame(data=np.array([[1., 2.], [3., 1.], [0., 2.]]),
                    columns=['N', 'Y'],
                    index=['H', 'L', 'M'])
ref5 = pd.DataFrame(data=np.array([[0., 2.], [3., 1.], [1., 2.]]),
                    columns=['N', 'Y'],
                    index=['M', 'L', 'H'])
ref6 = pd.DataFrame(data=np.array([[2., 1.], [1., 3.], [2., 0.]]),
                    columns=['Y', 'N'],
                    index=['H', 'L', 'M'])
ref7 = pd.DataFrame(data=np.array([[2., 0.], [1., 3.], [2., 1.]]),
                    columns=['Y', 'N'],
                    index=['M', 'L', 'H'])
@pytest.mark.parametrize('args, kwargs', [
    ([], {}),
    ([s1, s2_2], {}),
    ([1, 2, 3, 4, 5, 6, 7], {}),
    ([v1,v2], {'rev': 'bobafett'}),
    ([np.array([[[1,2],[3,4],[5,6], [1,2],[3,4],[5,6]]])], {})
    ])
def test_epitable_raise(args, kwargs):
    with pytest.raises(ValueError):
        epitable()
        
@pytest.mark.parametrize('args, kwargs, ref', [
    ([[88, 20, 555, 347]], {}, ref1),
    ([ref1], {}, ref1),
    ([s1, s2], {}, ref1),
    ([s1, s2.tolist()], {}, ref1),
    ([1, 2, 3, 4, 5, 6], {}, ref2),
    ([[1, 2, 3, 4, 5, 6]], {}, ref2),
    ([1, 2, 3, 4, 5, 6], {'ncol': 3}, ref3),
    ([[1, 2, 3, 4, 5, 6]], {'ncol': 3}, ref3),
    ([v1,v2], {'count': True}, ref4),
    ([v1,v2], {'count': True, 'rev': 'rows'}, ref5),
    ([v1,v2], {'count': True, 'rev': 'columns'}, ref6),
    ([v1,v2], {'count': True, 'rev': 'both'}, ref7)
    ])
def test_epitable(args, kwargs, ref):
    res = epitable(*args, **kwargs)
    assert_pd(res, ref)

    
    

