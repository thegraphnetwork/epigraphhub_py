"""
Last change on 2022/10/24
This module is used for fetching and downloading COVID
data from Federal Office of Public Health. The data
of interest consists in the following CSV tables:

_____________________________________________________________________
cases                  | COVID19Cases_geoRegion.csv
casesVaccPersons       | COVID19Cases_vaccpersons.csv
hosp                   | COVID19Hosp_geoRegion.csv
hospVaccPersons        | COVID19Hosp_vaccpersons.csv
death                  | COVID19Death_geoRegion.csv
deathVaccPersons       | COVID19Death_vaccpersons.csv
test                   | COVID19Test_geoRegion_all.csv
testPcrAntigen         | COVID19Test_geoRegion_PCR_Antigen.csv
hospCapacity           | COVID19HospCapacity_geoRegion.csv
hospCapacityCertStatus | COVID19HospCapacity_geoRegion_certStatus.csv
re                     | COVID19Re_geoRegion.csv
intCases               | COVID19IntCases.csv
virusVariantsWgs       | COVID19Variants_wgs.csv
covidCertificates      | COVID19Certificates.csv

Methods
-------

get_csv_relation(source):
    Generator which returns the context of interest for the foph data
    collection. Yields the table name and the respectively url.

download_csv(url):
    Runs curl in a url. Used for storing the CSVs in a properly directory.

remove_csvs():
    Removes the CSVs directory recursively.
"""
import os
import subprocess
from pathlib import Path

import requests
from loguru import logger

from epigraphhub.data._config import FOPH_CSV_PATH, FOPH_LOG_PATH, FOPH_URL

logger.add(FOPH_LOG_PATH, retention="7 days")


def fetch(source=FOPH_URL):
    """
    A generator responsible for accessing FOPH and retrieve the CSV
    relation, such as its Table name and URL as a tuple.

    Args:
        source (str) : The url with the csv relation.

    Returns:
        table (str)  : Table name as in the json file.
        url (str)    : URL to download the CSV.
    """
    context = requests.get(source).json()
    tables = context["sources"]["individual"]["csv"]["daily"]
    for table, url in tables.items():
        yield table, url


def download(url):
    """
    This methods runs curl in a URL that corresponds to a CSV file
    and stores it as specified in the URL.

    Args:
        url (str)    : URL that contains the CSV to download.
    """
    os.makedirs(FOPH_CSV_PATH, exist_ok=True)
    filename = url.split("/")[-1]
    subprocess.run(
        [
            "curl",
            "--silent",
            "-f",
            "-o",
            f"{FOPH_CSV_PATH}/{filename}",
            url,
        ]
    )
    logger.info(f"{filename} downloaded at {FOPH_CSV_PATH}.")


def remove(filename: str = None, entire_dir: bool = False):
    """
    Removes recursively the FOPH CSV's folder or filename.
    """
    if entire_dir:
        subprocess.run(["rm", "-rf", FOPH_CSV_PATH])
        logger.info(f"{FOPH_CSV_PATH} removed.")

    elif filename:
        file_path = Path(FOPH_CSV_PATH) / filename
        if file_path.exists():
            file_path.unlink()
            logger.info(f"{file_path} removed.")
        else:
            raise Exception(f"{file_path} not found.")

    else:
        logger.error(f"Set `entire_dir=True` to remove CSV dir")
        raise Exception("Nothing was selected to remove")
