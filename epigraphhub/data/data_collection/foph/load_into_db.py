import pandas as pd
from loguru import logger
from pangres import upsert
from epigraphhub.connection import get_engine
from epigraphhub.data.data_collection.config import (
    FOPH_LOG_PATH,
    FOPH_CSV_PATH,
)
from epigraphhub.settings import env

logger.add(FOPH_LOG_PATH, retention="7 days")


def load(table, filename):
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

