"""
Last change on 2022/10/24
This module is used for fetching and downloading COVID
data from Our World in Data. The data of interest consists in a
CSV table containing COVID information around the globe.


Methods
-------

download_csv():
    Runs curl from the OWID database and stores the CSV file in the tmp dir.

remove_csv():
    Removes the CSV file recursively.
"""
import os
import shlex as sx
import subprocess

from loguru import logger

from epigraphhub.connection import get_engine
from epigraphhub.data._config import (
    OWID_CSV_PATH,
    OWID_CSV_URL,
    OWID_FILENAME,
    OWID_HOST,
    OWID_LOG_PATH,
)
from epigraphhub.settings import env

logger.add(OWID_LOG_PATH, retention="7 days")
engine = get_engine(env.db.default_credential)


def download() -> None:
    """
    This method is responsible for download the CSV file from the
    OWID database. The file contains world information about COVID.
    """
    os.makedirs(OWID_CSV_PATH, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "--silent",
            "-f",
            "-o",
            f"{OWID_CSV_PATH}/{OWID_FILENAME}",
            f"{OWID_CSV_URL}",
        ]
    )
    logger.info("OWID csv downloaded.")


def compare() -> bool:
    table_size = _get_database_size(remote=False)
    csv_size = _get_csv_size()
    return table_size == csv_size


def remove() -> None:
    """
    This method deletes the OWID CSV file recursively.
    """
    os.remove(f"{OWID_CSV_PATH}/{OWID_FILENAME}")
    logger.info("OWID csv removed.")


def _get_database_size(remote=True) -> int:
    """
    Method responsible for connecting in the SQL database and return
    the total count of the OWID table.

    Args:
        remote (bool)         : If the SQL container is not locally configured,
                                creates a ssh tunnel with the Database.

    Raises:
        Exception (Exception) : Connection with the Database could not be
                                stablished.
    """
    if remote:
        proc = subprocess.Popen(
            sx.split(f"ssh -f epigraph@{OWID_HOST} -L 5432:localhost:5432 -NC")
        )
    try:
        engine = get_engine(env.db.default_credential)
        with engine.connect().execution_options(autocommit=True) as conn:
            curr = conn.execute("SELECT COUNT(*) FROM owid_covid")
            for count in curr:
                return int(count[0])
    except Exception as e:
        logger.error(f"Could not access OWID table\n{e}")
        raise (e)
    finally:
        if remote:
            proc.kill()


def _get_csv_size() -> int:
    """
    Method responsible for connecting in the SQL database and return
    the total count of the OWID CSV file.

    Raises:
        Exception (Exception) : CSV file not found.
    """
    try:
        raw_shape = subprocess.Popen(
            f"wc -l {os.path.join(OWID_CSV_PATH, OWID_FILENAME)}",
            shell=True,
            stdout=subprocess.PIPE,
        ).stdout
        clean = str(raw_shape.read()).split("'")
        shape = clean[1].split(" ")[0]
        return int(shape) - 1
    except Exception as e:
        logger.error(f"Could not reach OWID csv\n{e}")
