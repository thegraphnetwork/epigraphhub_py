from datetime import datetime

import pandas as pd
from loguru import logger

from epigraphhub.connection import get_engine
from epigraphhub.data.data_collection.config import FOPH_CSV_PATH, FOPH_LOG_PATH
from epigraphhub.settings import env

logger.add(FOPH_LOG_PATH, retention="7 days")


def csv_last_update(filename) -> datetime:
    df = pd.read_csv(f"{FOPH_CSV_PATH}/{filename}")
    if "date" not in df:
        last_update = df.datum.max()
    else:
        last_update = df.date.max()
    return datetime.strptime(str(last_update), "%Y-%m-%d")


def table_last_update(table) -> datetime:
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
