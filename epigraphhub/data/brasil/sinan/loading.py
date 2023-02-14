import os
from pathlib import Path

from loguru import logger
from pangres import upsert
from pysus.online_data import parquets_to_dataframe as to_df

from epigraphhub.connection import get_engine
from epigraphhub.data._config import SINAN_LOG_PATH
from epigraphhub.settings import env

logger.add(SINAN_LOG_PATH, retention="7 days")

engine = get_engine(credential_name=env.db.default_credential)


def upload(parquet_dirs: list):
    """
    Connects to the EGH SQL server and load all the chunks for all parquet
    directories extracted with `extract.download` into database.
    """

    for dir in parquet_dirs:
        if "parquet" in Path(dir).suffix and any(os.listdir(dir)):
            df = to_df(str(dir), clean_after_read=False)
            df.columns = df.columns.str.lower()
            df.index.name = "index"

            table_i = str(dir).split("/")[-1].split(".parquet")[0]
            table = table_i[:-4].lower()
            schema = "brasil"

            with engine.connect() as conn:
                try:

                    upsert(
                        con=conn,
                        df=df,
                        table_name=table,
                        schema=schema,
                        if_row_exists="update",
                        chunksize=1000,
                        add_new_columns=True,
                        create_table=True,
                    )

                    logger.info(f"Table {table} updated")

                except Exception as e:
                    logger.error(f"Not able to upsert {table} \n{e}")
                    raise e
