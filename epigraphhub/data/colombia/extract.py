"""
Last change on 2022/09/22
Comparing Colombia Governmental COVID data consists in a step before
pushing it to the SQL Database. Is responsible for retrieving the last
date in both CSV and SQL table. Connects to the Colombia data through
Socrata API and returns the maximum date found.

Methods
-------

web_last_update():
    Returns the max date from the Colombia data through Socrata API.

table_last_update():
    Connects to the Colombia SQL table and returns its max date.
"""
from datetime import datetime

from loguru import logger

from epigraphhub.connection import get_engine
from epigraphhub.data._config import COLOMBIA_CLIENT, COLOMBIA_LOG_PATH
from epigraphhub.settings import env

logger.add(COLOMBIA_LOG_PATH, retention="7 days")
client = COLOMBIA_CLIENT


def compare() -> bool:
    db_last_update = _table_last_update()
    data_last_update = _web_last_update()
    return db_last_update == data_last_update


def _table_last_update() -> datetime:
    """
    This method will connect to the SQL Database and query the maximum
    date found in Colombia table.

    Returns
    -------
    date : datetime
        Max date found in Colombia table.

    Raises
    ------
    Exception : Exception
        Unable to access Colombia table. @see epigraphhub.connection
    """
    engine = get_engine(credential_name=env.db.default_credential)
    try:
        with engine.connect().execution_options(autocommit=True) as conn:
            curr = conn.execute(
                "SELECT MAX(fecha_reporte_web) FROM colombia.positive_cases_covid_d"
            )
            for date in curr:
                date = dict(date)
                return date["max"]
    except Exception as e:
        logger.error(f"Could not access positive_cases_covid_d table\n{e}")
        raise (e)


def _web_last_update() -> datetime:
    """
    This method will request the maximum date found in Colombia data
    through Socrata API and returns it as a datetime object for further
    evaluation.

    Returns
    -------
    date : datetime
        Max date found in Colombia data through Socrata.

    Raises
    ------
    Exception : Exception
        Unable to create Socrata request.
    """
    try:
        report_date = [
            r for r in client.get_all("gt2j-8ykr", select="max(fecha_reporte_web)")
        ][0]
        last_update = datetime.strptime(
            report_date["max_fecha_reporte_web"], "%Y-%m-%d %H:%M:%S"
        )
        return last_update
    except Exception as e:
        logger.error(f"Could not access Socrata Api\n{e}")
        raise (e)
    finally:
        client.close()
