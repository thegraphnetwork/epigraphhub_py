import pandas as pd
import pytest

from epigraphhub.data.worldpop import WorldPop


def test_list_datasets():
    WP = WorldPop()
    ds = WP.datasets
    assert isinstance(ds, dict)
    assert len(WP.aliases) > 0


def test_print():
    WP = WorldPop()
    p = WP.__str__()
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
