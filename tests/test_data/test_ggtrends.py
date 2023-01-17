import pytest

from epigraphhub.data import ggtrends


def test_connect():
    ggtrends._get_connection()
    ggtrends._get_connection(language="ch-fr", timezone=60)


def test_payload():
    keywords = ["coronavirus", "covid"]
    trends = ggtrends._build_payload(keywords)


@pytest.mark.skip(reason="Google returned a response with code 500.")
def test_historical_interest():
    keywords = ["coronavirus", "covid"]
    df = ggtrends.historical_interest(keywords)
    assert not df.empty


def test_interest_over_time():
    keywords = ["coronavirus", "covid"]
    iot_df = ggtrends.interest_over_time(keywords)
    for k in keywords:
        assert k in iot_df.columns


@pytest.mark.skip(reason="Google returned a response with code 429.")
def test_interest_region():
    keywords = ["coronavirus", "covid"]
    df = ggtrends.interest_by_region(keywords, resolution="country", geo="CH")
    assert df.index.name == "geoName"


def test_related_topics():
    keywords = ["coronavirus", "covid"]
    d = ggtrends.related_topics(keywords)
    # assert len(d) > 0

@pytest.mark.skip(reason="Google returned a response with code 429.")
def test_related_queries():
    keywords = ["coronavirus", "covid"]
    d = ggtrends.related_queries(keywords)
    for kw in keywords:
        assert kw in d
    assert len(d) > 0
