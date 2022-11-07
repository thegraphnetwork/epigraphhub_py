"""
Last change on 2022/09/22
This module retrieves the data containing in the Our World in Data CSV file,
connects to the SQL Database and update it with the new data. Pangres
will generate chunks with total length of 10000, parse the date column to
datetime objects and insert the DataFrame into OWID SQL table.

Methods
-------

parse_types(df):
    Receives the DataFrame to parse the 'date' column values into
    datetime objects.

load(table, filename):
    Connects to SQL DB and update a table with CSV data using sqlalchemy.
    @see epigraphhub.connection
"""
import os
import shlex
import subprocess

import pandas as pd
from loguru import logger

from epigraphhub.connection import get_engine
from epigraphhub.data._config import (
    OWID_CSV_PATH,
    OWID_FILENAME,
    OWID_HOST,
    OWID_LOG_PATH,
)
from epigraphhub.settings import env

logger.add(OWID_LOG_PATH, retention="7 days")

engine_public = get_engine(env.db.default_credential)


def upload(remote=True):
    """
    A generator responsible connecting and loading data
    retrieved from the OWID CSV file into owid_covid table in
    the SQL Database, as defined in the connection configuration.

    Args:
        remote (bool)  : If the SQL container is not locally configured,
                         creates a ssh tunnel with the Database.
    """
    if remote:
        proc = subprocess.Popen(
            shlex.split(f"ssh -f epigraph@{OWID_HOST} -L 5432:localhost:5432 -NC")
        )
    try:
        data = pd.read_csv(os.path.join(OWID_CSV_PATH, OWID_FILENAME))
        data = _parse_date(data)
        engine = engine_public
        data.to_sql(
            "owid_covid",
            engine,
            index=False,
            if_exists="replace",
            method="multi",
            chunksize=10000,
        )
        logger.info("OWID data inserted into database")
    except Exception as e:
        logger.error(f"Could not update OWID table\n{e}")
        raise (e)
    finally:
        if remote:
            proc.kill()


def _parse_date(df):
    """
    Method responsible for receive the OWID DataFrame and return
    a DataFrame with the date parsed to datetime objects.

    Args:
        df (DataFrame) : OWID DataFrame.

    Returns:
        df (DataFrame) : OWID DataFrame with date column parsed to datetime.
    """
    df = df.convert_dtypes()
    df["date"] = pd.to_datetime(df.date)
    return df
