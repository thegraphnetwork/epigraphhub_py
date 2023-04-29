try:
    from typing import ParamSpecKwargs
except ImportError:
    # https://github.com/PyCQA/pylint/issues/5032#issuecomment-933886900
    from typing_extensions import ParamSpecKwargs

import pandas as pd
from pytrends.request import TrendReq


def _get_connection(
    language: str = "en-US", timezone: int = 360, **kwargs: ParamSpecKwargs
) -> TrendReq:
    return TrendReq(hl=language, tz=timezone, **kwargs)


def _build_payload(
    keywords: list[str], timeframe: str = "today 12-m", **kwargs: ParamSpecKwargs
) -> TrendReq:
    trends = _get_connection(**kwargs)
    trends.build_payload(keywords, cat=0, timeframe=timeframe, **kwargs)
    return trends


def historical_interest(
    keywords: list[str],
    year_start: int = 2021,
    month_start: int = 9,
    day_start: int = 1,
    hour_start: int = 0,
    year_end: int = 2021,
    month_end: int = 9,
    day_end: int = 30,
    hour_end: int = 0,
    cat: int = 0,
    sleep: int = 0,
    **kwargs: ParamSpecKwargs,
) -> pd.DataFrame:
    trends = _build_payload(keywords, **kwargs)
    df = trends.get_historical_interest(
        keywords,
        year_start=year_start,
        month_start=month_start,
        day_start=day_start,
        hour_start=hour_start,
        year_end=year_end,
        month_end=month_end,
        day_end=day_end,
        hour_end=hour_end,
        cat=cat,
        sleep=sleep,
    )
    return df


def interest_over_time(keywords: list[str], **kwargs: ParamSpecKwargs) -> pd.DataFrame:
    """
    Fetch trend time series for the `keywords` specified.
    """
    trends = _build_payload(keywords, **kwargs)
    interest_over_time_df = trends.interest_over_time()
    return interest_over_time_df


def interest_by_region(
    keywords: list[str], resolution: str = "country", **kwargs: ParamSpecKwargs
) -> pd.DataFrame:
    """
    Fetch trends by region

    Parameters
    ----------
    keywords : list
        List of keywords
    resolution : str, optional
        Spatial resolution can be one of ["country", "region", "city", "dma"].
        Defaults to "country".
    **kwargs : object

    Returns
    -------
    pd.DataFrame
    """
    trends = _build_payload(keywords, **kwargs)
    df = trends.interest_by_region(
        resolution=resolution.upper(), inc_low_vol=True, inc_geo_code=True
    )
    return df


def related_topics(keywords: list[str], **kwargs: ParamSpecKwargs) -> dict:
    """
    Get related topics to keywords provided
    Parameters
    ----------
    keywords : list
        List of keywords to find topics related to.
    **kwargs : object

    Returns
    -------
    dict
        Dictionary of DataFrames.
    """
    trends = _build_payload(keywords, **kwargs)
    dic: dict = trends.related_topics()
    return dic


def related_queries(keywords: list[str], **kwargs: ParamSpecKwargs) -> dict:
    """
    Get related queries to keywords provided.

    Parameters
    ----------
    keywords : list
        List of keywords to find queries related to.
    **kwargs : object

    Returns
    -------
    dict
        Dictionary of DataFrames.
    """
    trends = _build_payload(keywords, **kwargs)
    dic: dict = trends.related_queries()
    return dic
