from loguru import logger
from pysus.online_data import SINAN

from epigraphhub.data._config import PYSUS_DATA_PATH, SINAN_LOG_PATH

logger.add(SINAN_LOG_PATH, retention="7 days")


def download(disease: str):
    """
    Download all parquets available for a disease,
    according to `SINAN.agravos`.

    Attrs:
        disease (str): The disease to be downloaded.
    """

    SINAN.download_all_years_in_chunks(disease, data_dir=PYSUS_DATA_PATH)

    logger.info(f"All years for {disease} downloaded at {PYSUS_DATA_PATH}")
