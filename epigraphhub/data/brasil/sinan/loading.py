import os
from pathlib import Path

from loguru import logger
from pangres import upsert
from pysus.online_data import parquets_to_dataframe as to_df

from epigraphhub.connection import get_engine
from epigraphhub.data._config import PYSUS_DATA_PATH, SINAN_LOG_PATH
from epigraphhub.settings import env

logger.add(SINAN_LOG_PATH, retention="7 days")

engine = get_engine(credential_name=env.db.default_credential)


def upload():
    """
    Connects to the EGH SQL server and load all the chunks for all
    diseases found at `$PYSUS_DATA_PATH` into database. This method cleans
    the chunks left.

    """
    diseases_dir = Path(PYSUS_DATA_PATH).glob("*")
    di_years_dir = [x for x in diseases_dir if x.is_dir()]

    for dir in di_years_dir:
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
