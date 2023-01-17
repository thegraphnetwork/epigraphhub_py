"""
Created on Mon Jan 31 08:53:59 2022
@author: eduardoaraujo

Last change on 2022/09/22
This module is responsible retrieve generated chunks of data containing COVID
information collect via Socrata API from Colombia Governmental's data
collection. Connect to SQL Database and load chunks in order to update
positive_cases_covid_d table.

Methods
-------

gen_chunks_into_db():
    Generate chunks of data to insert into SQL Database using pangres.
"""
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger
from pangres import upsert

from epigraphhub.connection import get_engine
from epigraphhub.data._config import COLOMBIA_CLIENT, COLOMBIA_LOG_PATH
from epigraphhub.settings import env

logger.add(COLOMBIA_LOG_PATH, retention="7 days")
client = COLOMBIA_CLIENT


def upload():
    """
    This method will receive chunks generated by chunked_fetch and load them
    into the SQL Database. Pangres receives the records found in the Colombia data
    through Socrata API, uses the generator to load chunks with size of 10000 into
    SQL DB using upsert method.
    @note Colombia sometimes has a post update in the data, so rows update
          in this case is required to retrieve the rows updated.
    """
    slice_date = datetime.date(datetime.today()) - timedelta(200)
    slice_date = slice_date.strftime("%Y-%m-%d")

    # count the number of records that will be fetched
    records = client.get_all(
        "gt2j-8ykr",
        select="COUNT(*)",
        where=f'fecha_reporte_web > "{slice_date}"',
    )

    for i in records:
        record_count = i
        break

    del records

    maxrecords = int(record_count["COUNT"])

    engine = get_engine(env.db.default_credential)

    for df_new in _chunked_fetch(maxrecords):

        # save the data
        with engine.connect() as conn:
            upsert(
                con=conn,
                df=df_new,
                table_name="positive_cases_covid_d",
                schema="colombia",
                if_row_exists="update",
                chunksize=1000,
                add_new_columns=True,
                create_table=False,
            )

    logger.info("Table positive_cases_covid_d updated.")


def _chunked_fetch(maxrecords, start=0, chunk_size=10000):
    """
    Connects to Colombia database through Socrata API and generates
    slices of data in chunks in order to insert them into
    positive_cases_covid_d table. Updates the different values into
    a pattern to be easily queried.

    Args:
        maxrecords (int)   : Total rows count in the Colombia data.
        start (int)        : Parameter used to delimit the start of the
                             records in Colombia data.
        chunk_size (int)   : Size of the chunk to be inserted into SQL DB.

    Yields:
        df_new (DataFrame) : Dataframe with updated rows of fixed size.
    """
    slice_date = datetime.date(datetime.today()) - timedelta(200)

    slice_date = slice_date.strftime("%Y-%m-%d")

    while start < maxrecords:

        # Fetch the set of records starting at 'start'
        # create a df with this chunk files
        df_new = pd.DataFrame.from_records(
            client.get(
                "gt2j-8ykr",
                offset=start,
                limit=chunk_size,
                order="fecha_reporte_web",
                where=f'fecha_reporte_web > "{slice_date}"',
            )
        )

        df_new = df_new.rename(columns=str.lower)

        if df_new.empty:
            break

        df_new.set_index(["id_de_caso"], inplace=True)

        df_new = df_new.convert_dtypes()

        # change some strings to a standard
        df_new.replace(
            to_replace={
                "ubicacion": {"casa": "Casa", "CASA": "Casa"},
                "estado": {"leve": "Leve", "LEVE": "Leve"},
                "sexo": {"f": "F", "m": "M"},
            },
            inplace=True,
        )

        # transform the datetime columns in the correct time
        for c in df_new.columns:
            if c.lower().startswith("fecha"):
                df_new[c] = pd.to_datetime(df_new[c], errors="coerce")

        # eliminate any space in the end and start of the string values
        for i in df_new.select_dtypes(include=["string"]).columns:
            df_new[i] = df_new[i].str.strip()

        # Move up the starting record
        start = start + chunk_size

        yield df_new