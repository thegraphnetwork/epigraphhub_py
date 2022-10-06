import pandas
import pytest
from numpy import empty

from epigraphhub.data import worldbank as wbg


@pytest.mark.skip(reason="work in progress")
@pytest.mark.parametrize(
    "country, fx_et", [("BRA", "5Y"), ("BRA", "IN"), ("BRA", "TOTL")]
)
def test_get_pop_data(country, fx_et):

    df = wbg.get_pop_data(country, time=range(2015, 2021), fx_et=fx_et)

    assert df.empty == False


@pytest.mark.skip(reason="work in progress")
@pytest.mark.parametrize("keyword", [("pop"), ("all")])
def test_search_in_database(keyword):

    df = wbg.search_in_database(keyword)

    assert type(df) == pandas.core.frame.DataFrame


@pytest.mark.skip(reason="work in progress")
@pytest.mark.parametrize("keyword,db", [("pop", 2), ("AIDS", 16)])
def test_search_in_indicators(keyword, db):

    df = wbg.search_in_indicators(keyword, db)

    assert type(df) == pandas.core.frame.DataFrame


@pytest.mark.skip(reason="work in progress")
@pytest.mark.parametrize(
    "ind,country, db",
    [
        (["SP.POP.TOTL.FE.IN", "SP.POP.TOTL.MA.IN"], ["BRA", "USA"], 2),
        (["SP.POP.TOTL.MA.IN"], ["BRA"], 2),
        (["CPTOTNSXN"], ["USA", "CHE"], 15),
    ],
)
def test_get_world_data(ind, country, db):

    df = wbg.get_worldbank_data(ind, country, db)

    assert type(df.index) == pandas.core.indexes.datetimes.DatetimeIndex
    assert len(df.columns) == len(ind) + 2
    assert len(df.country.unique()) == len(country)
