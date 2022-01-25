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
