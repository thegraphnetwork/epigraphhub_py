from sodapy import Socrata

COLOMBIA_LOG_PATH = "/var/log/colombia_fetch.log"
COLOMBIA_CLIENT = Socrata("www.datos.gov.co", "078u4PCGpnDfH157kAkVFoWea")

FOPH_LOG_PATH = "/var/log/foph_fetch.log"
FOPH_URL = "https://www.covid19.admin.ch/api/data/context"
FOPH_CSV_PATH = "/tmp/foph/releases"

OWID_LOG_PATH = "/var/log/owid_fetch.log"
OWID_CSV_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
OWID_CSV_PATH = "/tmp/owid/releases"
OWID_FILENAME = OWID_CSV_URL.split("/")[-1]
OWID_HOST = "135.181.41.20"
