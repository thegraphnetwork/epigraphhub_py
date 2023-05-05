"""
Last change on 2023/04/17
This module is used for fetching and downloading COVID
data from Federal Office of Public Health. The data
of interest consists in the following CSV tables for daily
data:

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

get_csv_relation(source, freq, by):
    Generator which returns the context of interest for the foph data
    collection. Yields the table name and the respectively url.

download_csv(url):
    Runs curl in a url. Used for storing the CSVs in a properly directory.

remove_csvs():
    Removes the CSVs directory recursively.
"""
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

from epigraphhub.data._config import (
    FOPH_CSV_PATH,
    FOPH_LOG_PATH,
    FOPH_METADATA_URL,
    FOPH_URL,
)

logger.add(FOPH_LOG_PATH, retention="7 days")


def metadata(
    table: str = None, filename: str = None, source: str = FOPH_METADATA_URL
) -> pd.DataFrame:
    """
    Extracts metadata information from FOPH datasets. It is able
    of retrieving by table name or file name. Returns the relation
    of all tables if both are None.

    Args:
        table (str)      : The name of the table.
        filename (str)   : The name of the file of the table.
        source (str)     : The url where the metadata are stored.

    Returns:
        pd.DataFrame     : Returns a metadata DataFrame, if neither table or filename
                           are passed in, returns all tables available in source.
    """
    resp = requests.get(source).text
    soup = BeautifulSoup(resp)
    html_table = soup.find("table")
    table_rows = html_table.find_all("tr")

    metadatas = []
    for tr in table_rows:
        td = tr.find_all("td")
        row = [tr.text.strip() for tr in td if tr.text.strip()]
        if row:
            file, dataset, desc = row

            deprec = False
            if "DEPRECATED" in file:
                deprec = True
                file = str(file).split("DEPRECATED ")[1]

            file = str(file).split(".(json/csv)")[0]

            metadatas.append(
                dict(
                    table=dataset,
                    filename=file,
                    description=desc,
                    deprecated=deprec,
                )
            )

    df = pd.DataFrame(metadatas)

    if not table and not filename:
        return df

    hrefs = html_table.find_all("a", href=True)

    metadata_link = dict()
    for tag in hrefs:
        metadata_link[tag.text] = tag["href"]

    with open(f"{Path(__file__).parent}/schema.json") as schemas:
        data = json.load(schemas)

    table_schema = pd.DataFrame(data)

    if table:
        return pd.DataFrame(table_schema[table])

    if filename:
        return pd.DataFrame(
            table_schema[df[df["filename"] == filename].table.values[0]]
        )


def fetch(source: str = FOPH_URL, freq: str = "daily", by: str = "default") -> tuple:
    """
    A generator responsible for accessing FOPH and retrieve the CSV
    relation, such as its Table name and URL as a tuple.

    Parameters
    ----------
    source : str
        The url with the csv relation.
    freq : str
        The frequency of the data (daily or weekly).
    by : str
        Available only for weekly data, fetches cases by age,
        sex or default.

    Returns
    -------
    table : str
        Table name as in the json file.
    url : str
        URL to download the CSV.
    """
    context = requests.get(source).json()
    tables = context["sources"]["individual"]["csv"][freq]
    if freq.lower() == "weekly":
        if by.lower() == "age":
            tables = tables["byAge"]
        elif by.lower() == "sex":
            tables = tables["bySex"]
        else:
            tables = tables[by]
    for table, url in tables.items():
        yield table, url


def download(url):
    """
    This methods runs curl in a URL that corresponds to a CSV file
    and stores it as specified in the URL.

    Parameters
    ----------
    url : str
        URL that contains the CSV to download.
    """
    os.makedirs(FOPH_CSV_PATH, exist_ok=True)
    filename = url.split("/")[-1]
    filepath = Path(FOPH_CSV_PATH) / filename
    subprocess.run(
        [
            "curl",
            "--silent",
            "-f",
            "-o",
            f"{str(filepath)}",
            url,
        ]
    )
    logger.info(f"{filename} downloaded at {FOPH_CSV_PATH}.")
    return str(filepath)


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
