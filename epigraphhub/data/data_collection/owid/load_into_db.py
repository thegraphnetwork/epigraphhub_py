
import pandas as pd
import os
import shlex
import subprocess
from loguru import logger
from sqlalchemy import create_engine
from epigraphhub.settings import env
from epigraphhub.data.data_collection.config import (
    OWID_CSV_PATH,
    OWID_FILENAME,
    OWID_LOG_PATH,
    OWID_HOST,
)


with env.db.credentials[env.db.default_credential] as credential:
    engine_public = create_engine(
        f"postgresql://{credential.username}:"
        f"{credential.password}@{credential.host}:{credential.port}/"
        f"{credential.dbname}"
    )


logger.add(OWID_LOG_PATH, retention="7 days")


def parse_types(df):
    df = df.convert_dtypes()
    df["date"] = pd.to_datetime(df.date)
    logger.warning("OWID data types parsed.")
    return df


def load(remote=True):
    if remote:
        proc = subprocess.Popen(
            shlex.split(
                f"ssh -f epigraph@{OWID_HOST} -L 5432:localhost:5432 -NC"
            )
        )
    try:
        data = pd.read_csv(os.path.join(OWID_CSV_PATH, OWID_FILENAME))
        data = parse_types(data)
        engine = engine_public
        data.to_sql(
            "owid_covid",
            engine,
            index=False,
            if_exists="replace",
            method="multi",
            chunksize=10000,
        )
        logger.warning("OWID data inserted into database")
        with engine.connect() as connection:
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS country_idx 
                ON owid_covid (location);
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS iso_idx  ON owid_covid (iso_code);"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS date_idx ON owid_covid (date);"
            )
        logger.warning("Database indices created on OWID table")
    except Exception as e:
        logger.error(f"Could not update OWID table\n{e}")
        raise (e)
    finally:
        if remote:
            proc.kill()
