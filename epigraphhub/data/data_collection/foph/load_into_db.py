"""
Last change on 2022/09/22
This module will retrieve the data from a CSV file, create a connection
with the SQL Database and update it with the new information. Pangres
will generate chunks with total lenth of 1000 and insert them into the
corresponding table as specified by the downloaded CSV file.
@see epigraphhub.data.data_collection.foph.download
@see epigraphhub.connection

Methods
-------

load(table, filename):
    Connects to SQL DB and update a table with CSV information.
"""
import pandas as pd
from loguru import logger
from pangres import upsert

from epigraphhub.connection import get_engine
from epigraphhub.data.data_collection.config import FOPH_CSV_PATH, FOPH_LOG_PATH
from epigraphhub.settings import env

logger.add(FOPH_LOG_PATH, retention="7 days")


def load(table, filename):
    """
    A generator responsible connecting and loading data
    retrieved from a CSV file into its respective table in
    the SQL Database, as defined in the connection configuration.
    @see epigraphhub.connection

    Args:
        table (str)    : Raw table name. Later parsed to SQL format.
        filename (str) : File name as defined in the CSV URL.
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
            table_name=f"foph_{table.lower()}_d",
            schema="switzerland",
            if_row_exists="update",
            chunksize=1000,
            add_new_columns=True,
            create_table=True,
        )
    logger.info(f"Table foph_{table.lower()}_d updated")
