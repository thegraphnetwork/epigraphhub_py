"""
Last change on 2022/09/22
This module is responsible for including missing index in the FOPH
SQL tables. Connects to the SQL server as defined in the connection.
@see epigraphhub.connection

Methods
-------

parse_date_region(table):
    Connects to SQL DB and insert indexes if they are missing.
"""
from loguru import logger

from epigraphhub.connection import get_engine
from epigraphhub.settings import env


def parse_date_region(table):
    """
    Connects to the SQL DB and insert geoRegion and date indexes
    if they are missing.

    Raises
    ------
    Exception : Exception
        Unable to create index. Bad connection config.
    """
    engine = get_engine(env.db.default_credential)

    with engine.connect() as connection:
        try:
            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS region_idx
                ON switzerland.foph_{table.lower()} (geoRegion);
                """
            )
            logger.info(f"geoRegion index updated on foph_{table.lower()}")
        except Exception as e:
            logger.error(f"Could not create region index: {e}")
        try:
            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS date_idx
                ON switzerland.foph_{table.lower()} (date);
                """
            )
            logger.info(f"date index updated on foph_{table.lower()}")
        except Exception as e:
            logger.info(f"Could not create date index: {e}")
