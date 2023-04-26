"""
This module contains global variables used in data_collection for retrieving
COVID data, such as API connections, URLs for requests and files location,
commonly used in data collection modules
"""

from pathlib import Path

# Colombia COVID data config:
from sodapy import Socrata

COLOMBIA_LOG_PATH = "/tmp/colombia_fetch.log"
COLOMBIA_CLIENT = Socrata("www.datos.gov.co", "078u4PCGpnDfH157kAkVFoWea")


# Federal Office of Public Health (FOPH) COVID data config:
FOPH_LOG_PATH = "/tmp/foph_fetch.log"
FOPH_URL = "https://www.covid19.admin.ch/api/data/context"
FOPH_CSV_PATH = "/tmp/foph/releases"
FOPH_METADATA_URL = "https://www.covid19.admin.ch/api/data/documentation"


# Our World in Data (OWID) COVID data config:
OWID_LOG_PATH = "/tmp/owid_fetch.log"
OWID_CSV_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
OWID_CSV_PATH = "/tmp/owid/releases"
OWID_FILENAME = OWID_CSV_URL.split("/")[-1]
OWID_HOST = "135.181.41.20"


# SINAN data config:
SINAN_LOG_PATH = "/tmp/sinan_fetch.log"
_sinan_data = Path("/tmp") / "pysus"
_sinan_data.mkdir(exist_ok=True)
PYSUS_DATA_PATH = str(_sinan_data)
