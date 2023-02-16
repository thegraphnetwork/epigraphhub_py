import os
from pathlib import Path

import pandas as pd
from loguru import logger
from pangres import upsert

from epigraphhub.connection import get_engine
from epigraphhub.data._config import SINAN_LOG_PATH
from epigraphhub.settings import env

from . import DISEASES, normalize_str

logger.add(SINAN_LOG_PATH, retention="7 days")
engine = get_engine(credential_name=env.db.default_credential)


def upload(parquet_dirs: list):
    """
    Connects to the EpiGraphHub SQL server and load parquet chunks within
    directories, extracted using `extract.download`, into database. Receives
    a list of parquets directories with their full path, extract theirs
    DataFrames and upsert rows to Postgres connection following EGH table
    convention, see more in EGH's documentation:
    https://epigraphhub.readthedocs.io/en/latest/instruction_name_tables.html#about-metadata-tables
    Usage:
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

            df = _read_parquets_dir(dir)
            disease_code = str(dir).split("/")[-1].split(".parquet")[0][:-4]
            tablename = "sinan_" + normalize_str(di_codes[disease_code]) + "_m"
            schema = "brasil"

            print(f"Inserting {dir} on {schema}.{tablename}")

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


def _read_parquets_dir(path: str) -> pd.DataFrame:
    chunks = Path(path).glob("*.parquet")
    chunks_dfs = list()

    try:
        for parquet in chunks:
            df = pd.read_parquet(str(parquet), engine="fastparquet")
            objs = df.select_dtypes(object)
            df[objs.columns] = objs.apply(lambda x: x.str.replace("\x00", ""))
            chunks_dfs.append(df)

        final_df = pd.concat(chunks_dfs, ignore_index=True)
        final_df.columns = df.columns.str.lower()
        final_df.index.name = "index"

        return final_df

    except Exception as e:
        logger.error(e)
        raise e
