"""
This module contains global variables used in data_collection for retrieving
COVID data, such as API connections, URLs for requests and files location,
commonly used in data collection modules
"""

# Colombia COVID data config:
from sodapy import Socrata

COLOMBIA_LOG_PATH = "/var/log/colombia_fetch.log"
COLOMBIA_CLIENT = Socrata("www.datos.gov.co", "078u4PCGpnDfH157kAkVFoWea")


# Federal Office of Public Health (FOPH) COVID data config:
FOPH_LOG_PATH = "/var/log/foph_fetch.log"
FOPH_URL = "https://www.covid19.admin.ch/api/data/context"
FOPH_CSV_PATH = "/tmp/foph/releases"


# Our World in Data (OWID) COVID data config:
OWID_LOG_PATH = "/var/log/owid_fetch.log"
OWID_CSV_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
OWID_CSV_PATH = "/tmp/owid/releases"
OWID_FILENAME = OWID_CSV_URL.split("/")[-1]
OWID_HOST = "135.181.41.20"


# SINAN data config:
SINAN_LOG_PATH = "/var/log/sinan_fetch.log"
