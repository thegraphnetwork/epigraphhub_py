import os
from pathlib import Path

from loguru import logger
from pangres import upsert
from pysus.online_data import parquets_to_dataframe as to_df

from epigraphhub.connection import get_engine
from epigraphhub.data._config import SINAN_LOG_PATH
from epigraphhub.settings import env

from . import DISEASES, normalize_str

logger.add(SINAN_LOG_PATH, retention="7 days")

engine = get_engine(credential_name=env.db.default_credential)


def upload(parquet_dirs: list):
    """
    Connects to the EpiGraphHub SQL server and load all the chunks for all parquet
    directories extracted with `extract.download` into database. Receives
    a list of parquets directories with their full path, extract theirs
    DataFrames with `pysus.online_data.parquets_to_dataframe()` and upsert
    rows to Postgres connection following EGH table convention, see more:
    https://epigraphhub.readthedocs.io/en/latest/instruction_name_tables.html#about-metadata-tables

    Usage
    ```
    upload([
        '/tmp/pysus/ZIKABR17.parquet',
        '/tmp/pysus/ZIKABR18.parquet',
        '/tmp/pysus/ZIKABR19.parquet',
    ])
    """
    di_codes = {code: name for name, code in DISEASES.items()}

    for dir in parquet_dirs:
        if "parquet" in Path(dir).suffix and any(os.listdir(dir)):

            df = to_df(str(dir), clean_after_read=False)

            df.columns = df.columns.str.lower()
            df.index.name = "index"
            disease_code = str(dir).split("/")[-1].split(".parquet")[0][:-4]
            tablename = "sinan_" + normalize_str(di_codes[disease_code]) + "_m"
            schema = "brasil"

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

                    logger.info(f"Table {tablename} updated")

                except Exception as e:
                    logger.error(f"Not able to upsert {tablename} \n{e}")
                    raise e
