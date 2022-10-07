"""
Last change on 2022/09/22
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
import subprocess

from loguru import logger

from epigraphhub.data.data_collection.config import (
    OWID_CSV_PATH,
    OWID_CSV_URL,
    OWID_FILENAME,
    OWID_LOG_PATH,
)

logger.add(OWID_LOG_PATH, retention="7 days")


def download_csv():
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


def remove_csv():
    """
    This method deletes the OWID CSV file recursively.
    """
    os.remove(f"{OWID_CSV_PATH}/{OWID_FILENAME}")
    logger.info("OWID csv removed.")
