from loguru import logger

from epigraphhub.connection import get_engine
from epigraphhub.settings import env


def parse_date_region(table):
    engine = get_engine(env.db.default_credential)

    with engine.connect() as connection:
        try:
            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS region_idx
                ON switzerland.foph_{table.lower()}_d ('geoRegion');
                """
            )
            logger.info(f"geoRegion index updated on foph_{table.lower()}_d")
        except Exception as e:
            logger.error(f"Could not create region index: {e}")
        try:
            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS date_idx
                ON switzerland.foph_{table.lower()}_d (date);
                """
            )
            logger.info(f"date index updated on foph_{table.lower()}_d")
        except Exception as e:
            logger.info(f"Could not create date index: {e}")
