import time
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras as extras
import pyarrow.parquet as pq
from loguru import logger
from pandas.io.sql import get_schema
from pysus.online_data import SINAN
from sqlalchemy import create_engine

from epigraphhub.settings import env

logger.add("/tmp/pysus_insertions.log", retention="7 days")


# Database config
with env.db.credentials[env.db.default_credential] as credential:
    uri = (
        f"postgresql://{credential.username}:"
        f"{credential.password}@{credential.host}:{credential.port}/"
        f"{credential.dbname}"
    )
    engine = create_engine(uri)
    conn = psycopg2.connect(uri)


# Start measure of execution time
st = time.time()


def download_data(disease: str, year: int) -> str:
    """
    Download dataset and check if parquet directory exists.

    Parameters
    ----------

    disease: str
        Disease name.
    year: int
        Available year.

    Returns
    -------
    fname : str
        Name of parquet directory.
    """

    cod_agravo = SINAN.agravos.get(disease.title())

    fname = Path(f"{cod_agravo.upper()}BR{str(year)[2:]}.parquet/")

    if not fname.is_dir():
        elapsed_time = time.time() - st
        logger.info("Downloading data from the API...")
        fname = Path(SINAN.download(int(year), disease, return_fname=True))
        # Measure execution time ended
        elapsed_time = time.time() - st
        logger.info(
            "Get data from SUS to {}, finished in: {}".format(
                fname, time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            )
        )

    return fname


def parquet_to_df(fname: str) -> pd.DataFrame:
    """
    Convert the parquet files into a pandas DataFrame.

    Parameters
    ----------
    fname : str
        Name of the parquet files.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame.
    """

    df = (
        pq.ParquetDataset(
            f"{fname}/",
            use_legacy_dataset=False,
        )
        .read_pandas()  # columns=COL_NAMES
        .to_pandas()
    )

    # Measure execution time ended
    elapsed_time = time.time() - st
    logger.info(
        "Convert parquet files to dataFrame, finished in: {}".format(
            fname, time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        )
    )
    df.columns = df.columns.str.lower()

    return df.stack().str.decode("iso-8859-1").unstack()


def save_to_pgsql(disease: str, year: int) -> conn.commit:
    """
    Delete the table if it exists and create a new table according to the
    disease name and year.
    Adds DataFrame data to table.

    Parameters
    ----------
    fname: str
        Name of the parquet files.

    Returns
    -------
    conn: conn.commit
        Execute query with psycopg2.extras.
    """

    fname = download_data(disease, year)

    schema = "brasil"
    table = f"{str(fname)[:-8].lower()}"
    table_name = f'"{schema}"."{table}"'

    df = parquet_to_df(fname)
    query_sql = get_schema(df, table, schema=schema, con=engine)

    with conn.cursor(cursor_factory=extras.DictCursor) as cursor:
        logger.info(f"Creating columns and data types for {table_name} table")

        sql = f"""
            DROP TABLE IF EXISTS {table_name};
            {query_sql};
            """
        cursor.execute(sql)
        conn.commit()

    with conn.cursor(cursor_factory=extras.DictCursor) as cursor:
        logger.info(f"Inserting data into the {table_name} table...")
        tuples = [tuple(x) for x in df.to_numpy()]
        cols = ", ".join(list(df.columns))
        insert_sql = f"INSERT INTO {table_name}({cols}) VALUES %s"
        try:
            extras.execute_values(cursor, insert_sql, tuples)
            conn.commit()

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
            logger.info(e)
