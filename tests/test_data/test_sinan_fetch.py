import pathlib
import unittest

import pandas as pd
from pandas.io.sql import get_schema

from epigraphhub.data.sinan_fetch import (
    download_data,
    engine,
    parquet_to_df,
    save_to_pgsql,
)


class TestFethSinan(unittest.TestCase):
    def setUp(self):
        self.engine = engine
        self.disease = "zika"
        self.year = 2017
        self.fname = "ZIKABR17.parquet"
        self.table = "zikabr17"
        self.schema = "brasil"

    def test_download_data(self):
        _fname = download_data(self.disease, self.year)

        self.assertEqual(self.fname, str(_fname))
        self.assertIsInstance(_fname, pathlib.PosixPath)
        self.assertTrue(_fname.is_dir())

    def test_parquet_to_df(self):
        df = parquet_to_df(self.fname)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (32684, 38))

    def test_save_to_pgsql(self):
        save_to_pgsql(self.disease, self.year)
        table_name = f'"{self.schema}"."{self.table}"'
        query = "SELECT count(*) FROM brasil.zikabr17 LIMIT 10;"
        df1 = pd.read_sql(query, engine)
        df2 = parquet_to_df(self.fname)
        query_sql = get_schema(df2, self.table, schema=self.schema, con=engine)

        self.assertEqual(table_name, '"brasil"."zikabr17"')
        self.assertIn("CREATE TABLE brasil.zikabr17", query_sql)
        self.assertEqual(int(df1["count"]), df2.shape[0])
