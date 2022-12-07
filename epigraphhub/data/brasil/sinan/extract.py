from pathlib import PosixPath

from loguru import logger
from pysus.online_data import SINAN

from epigraphhub.data._config import SINAN_LOG_PATH

logger.add(SINAN_LOG_PATH, retention="7 days")

diseases = SINAN.agravos


def download(disease: str):
    """
    Download all parquets available for an disease,
    according to `SINAN.agravos`.

    Attrs:
        disease (str): The disease to be downloaded.
        data_dir (str) : The output directory were files will be downloaded.
                         A directory with the disease code will be created.

    Returns:
        parquets_paths_list list(PosixPath) : A list with all parquets dirs.
    """
    
    SINAN.download_all_years_in_chunks(disease)
    
    logger.info(f"All years for {disease} downloaded at /tmp/pysus")
