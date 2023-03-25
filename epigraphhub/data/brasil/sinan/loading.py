import os

from loguru import logger
from pangres import upsert
from pysus.online_data import parquets_to_dataframe

from epigraphhub.connection import get_engine
from epigraphhub.data._config import SINAN_LOG_PATH
from epigraphhub.settings import env

from . import normalize_str

logger.add(SINAN_LOG_PATH, retention="7 days")
engine = get_engine(credential_name=env.db.default_credential)


def upload(disease: str, parquet_dir: str) -> None:
    """
    Connects to the EpiGraphHub SQL server and load parquet chunks within
    directories, extracted using `extract.download`, into database. a local
    parquet dir (eg. ~/pysus/ZIKABR19.parquet), extract theirs DataFrames
    and upsert rows to Postgres connection following EGH table convention,
    see more in EGH's documentation:
    https://epigraphhub.readthedocs.io/en/latest/instruction_name_tables.html#about-metadata-tables
    """
    if any(os.listdir(parquet_dir)):
        df = parquets_to_dataframe(parquet_dir=parquet_dir)
        df.columns = df.columns.str.lower()
        df.index.name = "index"

        tablename = "sinan_" + normalize_str(disease) + "_m"
        schema = "brasil"
        print(f"Inserting {parquet_dir} on {schema}.{tablename}")

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
