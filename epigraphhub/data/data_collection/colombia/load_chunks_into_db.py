"""
Created on Mon Jan 31 08:53:59 2022

@author: eduardoaraujo
"""

from epigraphhub.data.data_collection.colombia.data_chunk import (
    DFChunkGenerator as gen,
)
from epigraphhub.data.data_collection.config import (
    COLOMBIA_LOG_PATH,
    COLOMBIA_CLIENT,
)
from epigraphhub.connection import get_engine
from datetime import datetime, timedelta
from pangres import upsert
from loguru import logger
from sqlalchemy import create_engine
from epigraphhub.settings import env

logger.add(COLOMBIA_LOG_PATH, retention="7 days")
client = COLOMBIA_CLIENT


def gen_chunks_into_db():

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

    start = 0
    chunk_size = 10000
    maxrecords = int(record_count["COUNT"])


    with env.db.credentials[env.db.default_credential] as credential:
        engine = create_engine(
            f"postgresql://{credential.username}:"
            f"{credential.password}@{credential.host}:{credential.port}/"
            f"{credential.dbname}"
        )

    for df_new in gen.chunked_fetch(start, chunk_size, maxrecords):

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
