#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:42:37 2022

@author: nr
"""
import numpy as np
from  scipy.stats import gamma, norm, poisson, hypergeom
from  scipy.stats import binomtest
from  scipy.optimize import brentq 
from typing import Optional, Union
from iteration_utilities import unique_everseen
import pandas as pd

###############################################################################
### UTILITIES
###############################################################################
def array_as_integer(arr):
    if arr.dtype in ['int32', 'int64']:
        return arr
    if arr.dtype in ['float32', 'float64']:
        is_int = np.vectorize(lambda x: x.is_integer(), otypes = [bool])
        if is_int(arr).all():
            return arr.astype('int')
        raise ValueError('Not all values in the array are integer!')
    try: 
        return arr.astype('int')
    except ValueError:
        raise ValueError('Not all values in the array are integer!')

def fisher_or_pval(x: np.array, or_: float = 1, alternative: str = 'two-sided'):
    m = x[:,0].sum()
    n = x[:,1].sum()
    k = x[0,:].sum()
    x = x[0,0]
    lo = max(0, k -n)
    hi = min(k,m)
    support = np.arange(lo, hi + 1)
    logdc = hypergeom.logpmf(support, m + n, m, k)
    
    def dnhyper(ncp):
        d = logdc + np.log(ncp)*support
        d = np.exp(d - d.max())
        return d / d.sum()
    
    def mnhyper(ncp):
        if ncp == 0:
            return lo
        if ncp == float('inf'):
            return hi
        return support*dnhyper(ncp)
    
    def pnhyper(q, ncp=1, upper_tail=False):
        if ncp == 1:
            return hypergeom.sf(x -1, m + n, m, k) if upper_tail else hypergeom.cdf(x,m + n, m, k)
        if ncp == 0:
            return int(q <= lo if upper_tail else q>= lo)
        if ncp == float('inf'):
            return int(q <= hi if upper_tail else q >= hi)
        return dnhyper(ncp)[support >= q if upper_tail else support <= q].sum()
      
    if alternative == 'two-sided':
        if or_ == 0:
            return int(x == lo) 
        if or_ == float('inf'):
            return int(x == hi)
        d = dnhyper(or_)
        return d[d <= d[x - lo]*(1 + 1e-7)].sum()
    elif alternative == 'less':
        return pnhyper(x, or_)
    elif alternative == 'greater':
        return pnhyper(x, or_, upper_tail=True)
    else:
        raise ValueError("`alternative` should be one of {'two-sided', 'less', 'greater'}")

###############################################################################
### epitools functions
###############################################################################
def ageadjust_direct(count: np.array, pop: np.array, stdpop: np.array, rate: Optional[np.array] = None, conf_level: float = 0.95) -> tuple:
    if rate is None:
        rate  = count/pop
    alpha = 1 - conf_level
    crude_rate = sum(count) / sum(pop)
    stdwt = stdpop/sum(stdpop)  # standard weight
    dsr = sum(stdwt * rate)  # directly standardised rate
    dsr_var = sum((stdwt**2) * (count/pop**2))
    wm = max(stdwt/pop)
    lci = gamma.ppf(alpha/2, (dsr**2)/dsr_var, scale=dsr_var/dsr)
    uci = gamma.ppf(1 - alpha/2, ((dsr+wm)**2)/(dsr_var+wm**2), scale=(dsr_var+wm**2)/(dsr+wm))
    return pd.Series({'crude_rate': crude_rate, 'adj_rate': dsr, 'lci': lci, 'uci': uci})

def ageadjust_indirect(count: np.array, pop: np.array, stdcount: np.array, stdpop: np.array, 
                       stdrate: Optional[np.array] = None, conf_level: float = 0.95):
    zv = norm.ppf(0.5*(1+conf_level))
    countsum = count.sum()
    if stdrate == None and len(stdcount) > 1 and len(stdpop) > 1:
        stdrate = stdcount/stdpop
    ##indirect age standardization
    ##a. sir calculation
    expected = (stdrate * pop).sum()
    sir = countsum/expected
    logsir_lci = np.log(sir) - zv * (1/np.sqrt(countsum))
    logsir_uci = np.log(sir) + zv * (1/np.sqrt(countsum))
    sir_lci = np.exp(logsir_lci)
    sir_uci = np.exp(logsir_uci)
    sir_series = pd.Series({'observed': countsum, 'exp': expected, 'sir': sir, 'lci': sir_lci, 'uci': sir_uci})
    ##b. israte calculation
    stdcrate = stdcount.sum()/stdpop.sum()
    crude_rate = count.sum()/pop.sum()
    isr = sir * stdcrate
    isr_lci = sir_lci * stdcrate
    isr_uci = sir_uci * stdcrate
    rate_series = pd.Series({'crude_rate': crude_rate, 'adj_rate': isr, 'lci': isr_lci, 'uci': isr_uci})
    return sir_series, rate_series  # NB. R returns a "list". It can be indexed by both labels and numbers. Should we reproduce that? It is far from standard Python behaviour

def binom_general(x: Union[list, tuple, np.array], n: Union[list, tuple, np.array],
                conf_level: float = 0.95, p: float = 0.5, method: str = 'exact'):  # Do we envision the case where conf_level is an array of different values?
    len_ = len(x)
    if type(x) != np.ndarray:
        x = np.array(x)
    if type(n) != np.ndarray:
        n = np.array(n)
    cis = np.array([list(binomtest(x[i], n[i], p=p).proportion_ci(confidence_level=conf_level, method=method)) for i in range(len_)])
    df_data = np.stack([x, n, x/n, cis[:, 0], cis[:,1], conf_level*np.ones(len_)], axis=1)
    return pd.DataFrame(data=df_data, columns=['x', 'n', 'proportion', 'lower', 'upper', 'conf_level'])

def binom_exact(x: Union[list, tuple, np.array], n: Union[list, tuple, np.array],
                conf_level: float = 0.95, p: float = 0.5):
    return binom_general(x, n, conf_level=conf_level, p=p, method='exact')

def binom_wilson(x: Union[list, tuple, np.array], n: Union[list, tuple, np.array],
                conf_level: float = 0.95, p: float = 0.5):
    return binom_general(x, n, conf_level=conf_level, p=p, method='wilson')

def binom_approx(x: Union[list, tuple, np.array], n: Union[list, tuple, np.array],
                conf_level: float = 0.95):
    if type(x) != np.ndarray:
        x = np.array(x)
    if type(n) != np.ndarray:
        n = np.array(n)
    Z = norm.ppf(0.5*(1+conf_level))
    SE_R = np.sqrt(x * (n - x) / (n**3))
    R_lci = x/n - Z*SE_R
    R_uci = x/n + Z*SE_R
    df_data = np.stack([x, n, x/n, R_lci, R_uci, conf_level*np.ones(len(x))],axis=1)
    return pd.DataFrame(data=df_data, columns=['x', 'n', 'proportion', 'lower', 'upper', 'conf_level'])

def kapmeier(time: Union[list, tuple, np.array], status: Union[list, tuple, np.array]):
    if type(time) != np.ndarray:
        time = np.array(time)
    if type(status) != np.ndarray:
        status = np.array(status)
    stime = sorted(time)
    status = status[np.argsort(time)]
    nj = np.arange(len(time), 0, -1)
    seen = set()
    o = [False if x in seen or seen.add(x) else True for x in stime]
    nj = nj[o]
    dj = pd.Series(status).groupby(pd.Series(stime)).sum()
    tj = list(unique_everseen(stime))
    sj = (nj - dj)/nj
    cumsj = sj.cumprod()
    cumrj = 1 - cumsj
    results = pd.DataFrame({'time': tj, 'n_risk': nj, 'n_events': dj,
                            'condsurv': sj, 'survival': cumsj, 'risk': cumrj})[dj != 0]
    return results.reset_index(drop=True)

def pois_exact(x: Union[list, tuple, np.array], pt: Union[int, float, list, tuple, np.array] = 1,
               conf_level: float = 0.95):  # Do we envision the case where conf_level is an array of different values?
    len_ = len(x)
    if type(x) != np.ndarray:
        x = np.array(x)
    if type(pt) in [list, tuple]:
        pt = np.array(pt)   
    elif type(pt) in [float, int]:
        pt = pt*np.ones(len_)
    results = np.zeros([len_, 6])
    f1 = lambda ans, x, alpha: poisson.cdf(x, ans) - 0.5*alpha
    f2 = lambda ans, x, alpha: 1 - poisson.cdf(x, ans) + poisson.pmf(x, ans) - 0.5*alpha
    for i in range(len_):
        alpha = 1 - conf_level
        a, b = [0, x[i]*5 + 4]  # interval
        assert f1(a, x[i], alpha)*f1(b, x[i], alpha) < 0, "f1(a) and f1(b) have the same sign"
        assert f2(a, x[i], alpha)*f2(b, x[i], alpha) < 0, "f2(a) and f2(b) have the same sign"
        # NB: in R "uniroot" is used, which, as brentq supposes f(a)*f(b)<0. If this can be assumed, maybe the two assertions are superfluous
        uci = brentq(f1, a, b, (x[i], alpha)) / pt[i]
        lci = brentq(f2, a, b, (x[i], alpha)) / pt[i] if x[i] else 0
        results[i] = np.array([x[i], pt[i], x[i] / pt[i], lci, uci, conf_level])
    return pd.DataFrame(data=results, columns=["x","pt","rate","lower","upper","conf_level"])

def pois_approx(x: Union[list, tuple, np.array], pt: Union[int, float, list, tuple, np.array] = 1,
               conf_level: float = 0.95):  # Do we envision the case where conf_level is an array of different values?
    len_ = len(x)    
    if type(x) != np.ndarray:
        x = np.array(x)
    if type(pt) in [list, tuple]:
        pt = np.array(pt)   
    elif type(pt) in [float, int]:
        pt = pt*np.ones(len_)
    Z = norm.ppf(0.5*(1+conf_level))
    SE_R = np.sqrt(x / pt**2)
    lower = x/pt - Z*SE_R
    upper = x/pt + Z*SE_R
    df = pd.DataFrame(data=np.stack([x, pt, x/pt, lower, upper], axis=1), 
                      columns = ['x', 'pt', 'rate', 'lower', 'upper'])
    df['conf_level'] = conf_level
    return df

def pois_daly(x: Union[list, tuple, np.array], pt: Union[int, float, list, tuple, np.array] = 1,
               conf_level: float = 0.95):  # Do we envision the case where conf_level is an array of different values?
    len_ = len(x)    
    if type(x) != np.ndarray:
        x = np.array(x)
    if type(pt) in [list, tuple]:
        pt = np.array(pt)   
    elif type(pt) in [float, int]:
        pt = pt*np.ones(len_)
    results = np.zeros([len_, 6])
    cipois = lambda xi, conf_level: (0, np.log(1 - conf_level)) if xi == 0 else\
        (gamma.ppf((1 - conf_level)/2, xi), gamma.ppf((1 + conf_level)/2, xi + 1))
    for i in range(len_):
        for_lci, for_uci = cipois(x[i], conf_level)
        results[i] = np.array([x[i], pt[i], x[i] / pt[i], for_lci/pt[i], for_uci/pt[i], conf_level])
    return pd.DataFrame(data=results, columns=["x","pt","rate","lower","upper","conf_level"])

def pois_byar(x: Union[list, tuple, np.array], pt: Union[int, float, list, tuple, np.array] = 1,
               conf_level: float = 0.95):  # Do we envision the case where conf_level is an array of different values?
    len_ = len(x)    
    if type(x) != np.ndarray:
        x = np.array(x)
    if type(pt) in [list, tuple]:
        pt = np.array(pt)   
    elif type(pt) in [float, int]:
        pt = pt*np.ones(len_)
    Z = norm.ppf(0.5*(1 + conf_level))
    aprime = x + 0.5
    Zinsert = (Z/3)*np.sqrt(1/aprime)
    lower = (aprime*(1-1/(9*aprime) - Zinsert)**3)/pt
    upper = (aprime*(1-1/(9*aprime) + Zinsert)**3)/pt
    df = pd.DataFrame(data=np.stack([x, pt, x/pt, lower, upper], axis=1), columns=["x","pt","rate","lower","upper"])
    df['conf_level'] = conf_level
    return df

def or_midp(x: Union[list, tuple, np.array], conf_level: float = 0.95,
            byrow: bool = True, interval: Union[list, tuple, np.array] = [0, 1000]):
    if type(x) != np.ndarray:
        x = np.array(x)
    x = array_as_integer(x)
    if x.shape == (4,):
        x = x.reshape(2,2) if byrow else x.reshape(2,2).T
    elif x.shape != (2,2):
        raise ValueError('x.shape must be either (4,) or ')
    mue = lambda x, or_: fisher_or_pval(x, or_=or_, alternative='less')\
        - fisher_or_pval(x, or_=or_, alternative='greater')
    midp = lambda x, or_: 0.5*(fisher_or_pval(x, or_=or_, alternative='less') - 
                               fisher_or_pval(x, or_=or_, alternative='greater') + 1)
    alpha = 1 - conf_level
    EST = brentq(lambda or_: mue(x, or_=or_), *interval)
    LCL = brentq(lambda or_: 1 - midp(x, or_=or_) - 0.5*alpha, *interval)
    if 0 in interval:
        i = interval.index(0)
        interval[i] += 1e-12
    UCL = 1/brentq(lambda or_: midp(x, or_=1/or_) - 0.5*alpha, *interval)
    d = {'x': x,  # returning function arguments is not customary in Python. Should we mirror R's implementation or follow pythonicness?
         'estimate': EST,
         'conf_int': (LCL, UCL),
         'conf_level': conf_level, # returning function arguments is not customary in Python. Should we mirror R's implementation or follow pythonicness?
         'method': 'median-unbiased estimate & mid-p exact CI'} # Such descriptive strings are not customary in Python. Should we mirror R's implementation or follow pythonicness?
    return d

def epitable(*args, count: bool = False, ncol: int = 2, byrow: bool = True, rev: str = ''):
    guessed = False
    if len(args) == 0:
        raise ValueError('No arguments provided')
    proc_array = False
    if len(args) == 1:  # 1 "vector"
        x = args[0]
        if type(x) in [tuple, list]:
            x = np.array(x)
        proc_array = True
        if type(x) == pd.DataFrame:
            df = x
            proc_array = False
    elif len(args) == 2:  # "2 vectors"
        x, y  = args
        type_x, type_y = type(x), type(y)
        if [type_x, type_y] == [pd.Series, pd.Series]:
            if (x.index != y.index).any():
                raise ValueError('the two series do not have the same index!')
            df = pd.concat([x,y], axis=1)
        elif pd.Series in [type_x, type_y]:  # a Series and something else
            (ser, nser, ser_first) = (x, y, True) if type_x == pd.Series else (y, x, False)
            if ser.name:
                (bname, num) = (ser.name[:-1], ser.name[-1] + 1 if ser_first else ser.name[-1] - 1)  if ser.name[-1].isnumeric() else (ser.name, 2 if ser_first else 1)
            else:
                ser.name = 'Disease1'
                (bname, num) = ('Disease', 2 if ser_first else 1)
                guessed = True
            nser = pd.Series(nser, index=ser.index, name=f'{bname}{num}')
            df = pd.concat([ser, nser] if ser_first else [nser, ser], axis=1)
        elif [x in [tuple,list] for x in [type_x, type_y]] == [True, True]:
            x, y = np.array(x), np.array(y)
            arr = np.stack([x,y], axis=1)
            df = pd.DataFrame(data=arr, index=[f'Exposed{i+1}' for i in range(len(x))], columns=['Disease1', 'Disease2'])
            guessed = True
    else:  # each number as an element in args
        x = np.array(args)
        proc_array = True
    if proc_array:
        if x.dtype in ['<U1', 'U21']:
            raise ValueError('Single character vector not allowed.')
        if len(x.shape) == 1:
            try:
                x = x.reshape(-1, ncol) if byrow else x.reshape(ncol, -1).T
            except ValueError:
                raise ValueError('number of elements is not a multiple of "ncol"')
        if len(x.shape) == 2:
            nr, nc = x.shape
            df = pd.DataFrame(data=x, columns=[f'Disease{i+1}' for i in range(nc)], index=[f'Exposed{i+1}' for i in range(nr)])
            guessed = True
        else:
            raise ValueError('Cannot interpret array with 3 or more axes')
    if count:
        df = df.groupby(df.columns.tolist()).size().unstack()
        if guessed:
            df.columns.name, df.index.name = None, None
    df.index.name = 'Predictor' if df.index.name is None else df.index.name
    df.columns.name = 'Outcome' if df.columns.name is None else df.columns.name
    if rev:
        if rev not in ['both', 'columns', 'rows']:
            raise ValueError('for "rev" use one of "both", "columns", "rows" to rev, leave default empty string not to reverse')
        slccols = slice(None, None, -1 if rev in ['columns', 'both'] else 1)
        slcrows = slice(None, None, -1 if rev in ['rows', 'both'] else 1)
        x = df.values[slcrows, slccols]
        df = pd.DataFrame(data=x, columns=df.columns[slccols], index=df.index[slcrows])
    return df
  
def table_margins(x: Union[np.array, pd.DataFrame]):
    has_labels = False
    if type(x) == pd.DataFrame:
        df, x = x, x.values
        has_labels = True
    nrows, ncols = x.shape
    tr = np.zeros([nrows + 1, ncols + 1])
    tr[:nrows, :ncols] = x
    tr[-1,:-1] = x.sum(axis=0)
    tr[:-1, -1] = x.sum(axis=1)
    tr[-1,-1] = x.sum()
    col_labels = df.columns.tolist() if has_labels else [f'col{i}' for i in range(ncols)]
    col_labels.append('Total')
    row_labels = df.index.tolist() if has_labels else [f'row{i}' for i in range(nrows)]
    row_labels.append('Total')
    return pd.DataFrame(tr, columns=col_labels, index=row_labels)
          

