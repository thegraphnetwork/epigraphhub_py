import os
from pathlib import Path

import pandas as pd
from pysus import SINAN
from loguru import logger
from pangres import upsert
from pysus.classes.sinan import Disease

from epigraphhub.connection import get_engine
from epigraphhub.data._config import SINAN_LOG_PATH, PYSUS_DATA_PATH
from epigraphhub.settings import env

from . import normalize_str

logger.add(SINAN_LOG_PATH, retention="7 days")
engine = get_engine(credential_name=env.db.default_credential)


def upload(disease: str, data_path: str = PYSUS_DATA_PATH):
    """
    Connects to the EpiGraphHub SQL server and load parquet chunks within
    directories, extracted using `extract.download`, into database. Receives
    a disease and look for local parquets paths in PYSUS_DATA_PATH, extract theirs
    DataFrames and upsert rows to Postgres connection following EGH table
    convention, see more in EGH's documentation:
    https://epigraphhub.readthedocs.io/en/latest/instruction_name_tables.html#about-metadata-tables
    """
    disease_years = Disease(disease).get_years(stage='all')

    for year in disease_years:
        df = SINAN.parquets_do_df(disease, year, data_path)
        df.columns = df.columns.str.lower()
        df.index.name = "index"

        tablename = "sinan_" + normalize_str(disease) + "_m"
        schema = "brasil"

        print(f"Inserting {disease}-{year} on {schema}.{tablename}")

        with engine.connect() as conn:
            try:
                upsert(
                    con=conn,
                    df=df,
                    table_name=tablename,
                    schema=schema,
                    if_row_exists="update",
                    chunksize=1000,
                    add_new_columns=True,
                    create_table=True,
                )

                print(f"Table {tablename} updated")

            except Exception as e:
                logger.error(f"Not able to upsert {tablename} \n{e}")
                raise e
