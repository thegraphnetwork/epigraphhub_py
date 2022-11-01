import time
from pathlib import Path

import pandas as pd
import psycopg2.extras as extras
from loguru import logger
from pandas.io.sql import get_schema

from epigraphhub.connection import get_engine
from epigraphhub.data._config import SINAN_LOG_PATH
from epigraphhub.settings import env

st = time.time()
logger.add(SINAN_LOG_PATH, retention="7 days")

engine = get_engine(credential_name=env.db.default_credential)


def upload(tablename: str, df: pd.DataFrame):
    """
    Delete the table if it exists and create a new table
    according to the disease name and year.
    Adds dataframe data to table.

    Parameters
    ----------
        fname: Name of the parquet files.
    Returns
    -------
        conn: Execute query with psycopg2.extras.
    """

    schema = "brasil"
    table_name = f'"{schema}"."{tablename}"'

    query_sql = get_schema(df, tablename, schema=schema, con=engine)

    with engine.connect().cursor(cursor_factory=extras.DictCursor) as cursor:
        logger.info(f"Creating columns and data types for {table_name} table")

        sql = f"""
            DROP TABLE IF EXISTS {table_name};
            {query_sql};
            """
        cursor.execute(sql)
        engine.connect().commit()

    with engine.connect().cursor(cursor_factory=extras.DictCursor) as cursor:
        logger.info(f"Inserting data into the {table_name} table...")
        tuples = [tuple(x) for x in df.to_numpy()]
        cols = ", ".join(list(df.columns))
        insert_sql = f"INSERT INTO {table_name}({cols}) VALUES %s"
        try:
            extras.execute_values(cursor, insert_sql, tuples)
            engine.connect().commit()

            # Measure execution time ended
            elapsed_time = time.time() - st

            logger.info(
                "Pysus: {} rows in {} fields inserted in the database, "
                "finished at: {}".format(
                    df.shape[0],
                    df.shape[1],
                    time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
                )
            )

        except Exception as e:
            logger.error(e)
