from pytrends.request import TrendReq


def _get_connection(language="en-US", timezone=360, **kwargs):
    return TrendReq(hl=language, tz=timezone)


def _build_payload(keywords: list[str], **kwargs) -> object:
    trends = _get_connection(**kwargs)
    trends.build_payload(keywords, cat=0, timeframe="today 12-m")
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
    **kwargs,
) -> object:
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


def interest_over_time(keywords: list[str], **kwargs):
    trends = _build_payload(keywords, **kwargs)
    interest_over_time_df = trends.interest_over_time()
    return interest_over_time_df


def interest_by_region(keywords: list[str], resolution: str = "country", **kwargs):
    trends = _build_payload(keywords, **kwargs)
    df = trends.interest_by_region(
        resolution=resolution, inc_low_vol=True, inc_geo_code=True
    )
    return df


def related_topics(keywords: list[str], **kwargs) -> dict:
    """
    Get related topics to keywords provided
    Args:
        keywords: list of keywords to find topics related to.
        **kwargs:

    Returns: dictionary of dataframes

    """
    trends = _build_payload(keywords, **kwargs)
    dic = trends.related_topics()
    return dic


def related_queries(keywords: list[str], **kwargs) -> dict:
    """
    Get related queries to keywords provided
    Args:
        keywords: list of keywords to find queries related to.
        **kwargs:

    Returns: dictionary of dataframes

    """
    trends = _build_payload(keywords, **kwargs)
    dic = trends.related_queries()
    return dic
