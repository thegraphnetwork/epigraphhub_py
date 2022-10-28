"""
Last change on 2022/10/24
This module is responsible for including missing index in the OWID
SQL table. Connects to the SQL server as defined in the connection.
@see epigraphhub.connection

Methods
-------

parse_indexes(table, remote):
    Connects to SQL DB and insert indexes if they are missing.
"""
import shlex
import subprocess

from loguru import logger

from epigraphhub.connection import get_engine
from epigraphhub.data._config import OWID_HOST, OWID_LOG_PATH
from epigraphhub.settings import env

logger.add(OWID_LOG_PATH, retention="7 days")


def parse_indexes(remote=True):
    """
    Connects to the SQL DB and insert location, iso_code and date indexes
    if they are missing.

    Args:
        remote (bool)         : If the SQL container is not locally configured,
                                creates a ssh tunnel with the Database.

    Raises:
        Exception (Exception) : Unable to create index. Bad connection config.
    """
    if remote:
        proc = subprocess.Popen(
            shlex.split(f"ssh -f epigraph@{OWID_HOST} -L 5432:localhost:5432 -NC")
        )
    engine = get_engine(env.db.default_credential)

    with engine.connect() as connection:
        try:
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS country_idx
                ON owid_covid (location);
                """
            )
            logger.info(f"location index updated on owid_covid")
        except Exception as e:
            logger.error(f"Could not create location index: {e}")
        try:
            connection.execute(
                "CREATE INDEX IF NOT EXISTS iso_idx ON owid_covid (iso_code);"
            )
        except Exception as e:
            logger.error(f"Could not create iso_code index: {e}")
        try:
            connection.execute(
                "CREATE INDEX IF NOT EXISTS date_idx ON owid_covid (date);"
            )
            logger.info(f"date index updated on owid_covid")
        except Exception as e:
            logger.info(f"Could not create date index: {e}")
        finally:
            logger.info("Database indexes created on OWID table")
            if remote:
                proc.kill()
