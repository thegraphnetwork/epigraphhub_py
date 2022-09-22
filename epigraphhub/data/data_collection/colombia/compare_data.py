from loguru import logger
from datetime import datetime
from epigraphhub.connection import get_engine
from epigraphhub.data.data_collection.config import (
    COLOMBIA_LOG_PATH,
    COLOMBIA_CLIENT,
)
from epigraphhub.settings import env

logger.add(COLOMBIA_LOG_PATH, retention="7 days")
client = COLOMBIA_CLIENT


def table_last_update() -> datetime:
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


def web_last_update() -> datetime:
    try:
        report_date = [
            r
            for r in client.get_all(
                "gt2j-8ykr", select="max(fecha_reporte_web)"
            )
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
