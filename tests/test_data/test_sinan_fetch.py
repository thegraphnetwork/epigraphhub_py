import os
import unittest
from pathlib import Path

import pandas as pd

from epigraphhub.connection import get_engine
from epigraphhub.data.brasil.sinan import extract, loading, viz
from epigraphhub.settings import env

engine = get_engine(credential_name=env.db.default_credential)


class TestFethSinan(unittest.TestCase):
    def setUp(self):
        self.engine = engine
        self.disease = "Zika"
        self.year = 2017
        self.fpath = ["/tmp/pysus/ZIKA/ZIKABR17.parquet"]
        self.table = "zika17"
        self.schema = "brasil"

    def test_download_data_zika(self):

        _fname = extract.download(self.disease)

        self.assertTrue(Path(self.fpath[0]).exists())
        self.assertTrue(any(os.listdir(self.fpath[0])))
        self.assertEqual(
            _fname,
            [
                "/tmp/pysus/ZIKA/ZIKABR16.parquet",
                "/tmp/pysus/ZIKA/ZIKABR17.parquet",
                "/tmp/pysus/ZIKA/ZIKABR18.parquet",
                "/tmp/pysus/ZIKA/ZIKABR19.parquet",
                "/tmp/pysus/ZIKA/ZIKABR20.parquet",
                "/tmp/pysus/ZIKA/ZIKABR21.parquet",
            ],
        )

    def test_parquet_visualization(self):

        df = viz.parquet(self.fpath[0], clean_after_read=False)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (32684, 38))

    def test_save_to_pgsql(self):

        loading.upload(self.fpath)

    def test_table_visualization(self):

        df = viz.table(self.disease, self.year)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
