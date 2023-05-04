"""
Last change on 2022/10/24
This module will retrieve the data from a CSV file, create a connection
with the SQL Database and update it with the new information. Pangres
will generate chunks with total length of 1000 and insert them into the
corresponding table as specified by the downloaded CSV file.
@see epigraphhub.connection

Methods
-------

load(table, filename):
    Connects to SQL DB and update a table with CSV information.
"""
from datetime import datetime

import pandas as pd
from loguru import logger
from pangres import upsert

from epigraphhub.connection import get_engine
from epigraphhub.data._config import FOPH_CSV_PATH, FOPH_LOG_PATH
from epigraphhub.settings import env

logger.add(FOPH_LOG_PATH, retention="7 days")


def upload(table, filename):
    """
    A generator responsible connecting and loading data
    retrieved from a CSV file into its respective table in
    the SQL Database, as defined in the connection configuration.
    @see epigraphhub.connection

    Parameters
    ----------
    table : str
        Raw table name. Later parsed to SQL format.
    filename : str
        File name as defined in the CSV URL.
        @see .download as above.
    """
    new_df = pd.read_csv(f"{FOPH_CSV_PATH}/{filename}")
    logger.info(f"Reading {filename}")

    new_df = new_df.rename(columns=str.lower)
    new_df.index.name = "id_"
    if "date" not in new_df.columns:
        new_df["date"] = pd.to_datetime(new_df.datum)
    else:
        new_df["date"] = pd.to_datetime(new_df.date)
    logger.info(f"Table {table} passed to DataFrame")

    engine = get_engine(env.db.default_credential)
    with engine.connect() as conn:
        upsert(
            con=conn,
            df=new_df,
            table_name=f"foph_{table.lower()}",
            schema="switzerland",
            if_row_exists="update",
            chunksize=1000,
            add_new_columns=True,
            create_table=True,
        )
    logger.info(f"Table foph_{table.lower()} updated")


def compare(filename, table) -> bool:
    csv_date = _csv_last_update(filename)
    table_date = _table_last_update(table)
    return csv_date == table_date


def _csv_last_update(filename) -> datetime:
    """
    Method responsible for retrieving the maximum date in a CSV file.

    Parameters
    ----------
    filename : str
        The CSV filename.

    Returns
    -------
    last_update : datetime
        Datetime with the max date in the CSV.

    Raises
    ------
    Exception : Exception
        Empty DataFrame from CSV.
    """
    df = pd.read_csv(f"{FOPH_CSV_PATH}/{filename}")
    if "date" not in df:
        last_update = df.datum.max()
    else:
        last_update = df.date.max()
    if df.empty:
        raise Exception("Empty file.")
    return datetime.strptime(str(last_update), "%Y-%m-%d")


def _table_last_update(table) -> datetime:
    """
    Method responsible for connecting and retrieving the maximum date of a table
    in the SQL Database. @see epigraphhub.connection : Where the connection is
    configured.

    Parameters
    -----------
    table : str
        Table name as in the CSV file. Later transformed into SQL DB table
        format.

    Returns
    -------
    last_update : datetime
        Datetime with the max date in the table.

    Raises
    ------
    Exception : Exception
        Connection with the database could not be established.
    """
    engine = get_engine(env.db.default_credential)
    try:
        df = pd.read_sql(f"select * from switzerland.foph_{table.lower()};", engine)
        if "date" not in df:
            df = df.datum.dropna()
            last_update = df.max()
        df = df.date.dropna()
        last_update = df.max()
        return datetime.strptime(str(last_update), "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"Could not access {table} table\n{e}")
        raise (e)
