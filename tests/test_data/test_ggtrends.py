import pytest

from epigraphhub.data import ggtrends


def test_connect():
    ggtrends._get_connection()
    ggtrends._get_connection(language="ch-fr", timezone=60)


def test_payload():
    keywords = ["coronavirus", "covid"]
    trends = ggtrends._build_payload(keywords)


def test_historical_interest():
    keywords = ["coronavirus", "covid"]
    df = ggtrends.historical_interest(keywords)
    assert len(df) > 0


def test_interest_over_time():
    keywords = ["coronavirus", "covid"]
    iot_df = ggtrends.interest_over_time(keywords)
    for k in keywords:
        assert k in iot_df.columns
