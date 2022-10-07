"""
Last change on 2022/09/22
Comparing Federal Office of Public Health (FOPH) COVID data consists in
a step before pushing it to the database. Is responsible for retrieving
the last date in both CSV and SQL table.

Methods
-------

csv_last_update(filename):
    Returns the max date in a CSV file.

table_last_update(table):
    Connects to the SQL database and returns its max date.
"""
from datetime import datetime

import pandas as pd
from loguru import logger

from epigraphhub.connection import get_engine
from epigraphhub.data.data_collection.config import FOPH_CSV_PATH, FOPH_LOG_PATH
from epigraphhub.settings import env

logger.add(FOPH_LOG_PATH, retention="7 days")


def csv_last_update(filename) -> datetime:
    """
    Method responsible for retrieving the maximum date in a CSV file.

    Args:
        filename (str)         : The CSV filename.

    Returns:
        last_update (datetime) : Datetime with the max date in the CSV.

    Raises:
        Exception (Exception)  : Empty dataframe from CSV.
    """
    df = pd.read_csv(f"{FOPH_CSV_PATH}/{filename}")
    if "date" not in df:
        last_update = df.datum.max()
    else:
        last_update = df.date.max()
    if df.empty:
        raise Exception("Empty file.")
    return datetime.strptime(str(last_update), "%Y-%m-%d")


def table_last_update(table) -> datetime:
    """
    Method responsible for connecting and retrieving the maximum date
    of a table in the SQL Database.
    @see epigraphhub.connection : Where the connection is configured.

    Args:
        table (str)             : Table name as in the CSV file. Later
                                  tansformed into SQL DB table format.

    Returns:
        last_update (datetime)  : Datetime with the max date in the table.

    Raises:
        Exception (Exception)   : Connection with the database could not be
                                  stablished.
    """
    engine = get_engine(env.db.default_credential)
    try:
        df = pd.read_sql(f"select * from switzerland.foph_{table.lower()}_d;", engine)
        if "date" not in df:
            df = df.datum.dropna()
            last_update = df.max()
        df = df.date.dropna()
        last_update = df.max()
        return datetime.strptime(str(last_update), "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"Could not access {table} table\n{e}")
        raise (e)
