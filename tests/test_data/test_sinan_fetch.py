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
        self.data_dir = Path.home() / 'pysus'
        self.file = ["ZIKABR17.parquet"]
        self.table = "zika"
        self.schema = "brasil"

    def test_download_data_zika(self):
        extract.download(self.disease)
        self.assertTrue(any(os.listdir(self.data_dir)))
        self.assertTrue(self.file[0] in os.listdir(self.data_dir))

    def test_parquet_visualization(self):
        fpath = Path(self.data_dir) / self.file[0]
        df = viz.parquet(fpath, clean_after_read=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (32684, 38))

    @unittest.skip("Need table to test")  # TODO: need table to test
    def test_save_to_pgsql(self):
        loading.upload(self.file[0])

    @unittest.skip("Need table to test")  # TODO: need table to test
    def test_table_visualization(self):
        df = viz.table(self.disease, self.year)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
