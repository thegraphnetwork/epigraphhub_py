import pandas as pd
import pytest

from epigraphhub.data.worldpop import WorldPop, json_get


def test_get_root():
    res = json_get("https://www.worldpop.org/rest/data")
    assert "data" in res
    assert isinstance(res, dict)


def test_list_datasets():
    WP = WorldPop()
    ds = WP.datasets
    assert isinstance(ds, dict)
    assert len(WP._aliases) > 0


def test_print():
    WP = WorldPop()

    p = WP.__repr__()

    assert p.startswith("|")
    assert "alias" in p
    assert "title" in p


def test_get_ds_tables():
    WP = WorldPop()
    for df in WP.get_dataset_tables("pop"):
        assert isinstance(df, dict)
        assert isinstance(df["data"], pd.DataFrame)
        assert "name" in df
        assert len(df["data"]) > 0


def test_get_data_country():
    WP = WorldPop()
    c = WP.get_data_by_country(alias="pop", level="pic", ISO3_code="BRA")
    assert isinstance(c, dict)
    assert "error" not in c
