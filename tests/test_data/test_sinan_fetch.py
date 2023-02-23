import os
import unittest
from pathlib import Path

import pandas as pd

from epigraphhub.connection import get_engine
from epigraphhub.data.brasil.sinan import DISEASES, extract, loading, normalize_str, viz
from epigraphhub.settings import env

engine = get_engine(credential_name=env.db.default_credential)


class TestFethSinan(unittest.TestCase):
    def setUp(self):
        self.engine = engine
        self.disease = "Zika"
        self.year = 2017
        self.data_dir = Path("/tmp") / "pysus"
        self.file = ["ZIKABR17.parquet"]
        self.table = "sinan_zika_m"
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

    def test_metadata_extraction(self):
        anim_metadata = extract.metadata_df('Animais Peçonhentos')
        self.assertTrue(isinstance(anim_metadata, pd.DataFrame))
        self.assertEqual(anim_metadata.shape, (58, 7))
        self.assertEqual(
            list(anim_metadata.columns),
            [
                'Nome do campo',
                'Campo',
                'Tipo',
                'Categoria',
                'Descrição',
                'Características',
                'DBF'
            ]
        )


    @unittest.skip("Need table to test")  # TODO: need table to test
    def test_save_to_pgsql(self):
        loading.upload(self.file[0])

    @unittest.skip("Need table to test")  # TODO: need table to test
    def test_table_visualization(self):
        df = viz.table(self.disease, self.year)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)


class TestSINANUtilities(unittest.TestCase):
    def test_diseases_dictionary(self):
        self.assertTrue("Animais Peçonhentos" in DISEASES.keys())
        self.assertTrue("DENG" in DISEASES.values())
        self.assertEqual("ZIKA", DISEASES["Zika"])

    def test_normalizing_disease_name(self):
        d1 = "Sífilis Adquirida"
        d2 = "Animais Peçonhentos"
        d3 = "Contact Communicable Disease"

        norm = lambda d: normalize_str(d)
        self.assertEqual(norm(d1), "sifilis_adquirida")
        self.assertEqual(norm(d2), "animais_peconhentos")
        self.assertEqual(norm(d3), "contact_communicable_disease")
