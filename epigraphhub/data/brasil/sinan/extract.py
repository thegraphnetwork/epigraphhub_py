import os
from pathlib import Path

import pandas as pd
from loguru import logger
from pysus import SINAN

from epigraphhub.data._config import PYSUS_DATA_PATH, SINAN_LOG_PATH

logger.add(SINAN_LOG_PATH, retention="7 days")


def download(disease: str, years: list = None) -> None:
    """
    Download all parquets available for a disease,
    according to `SINAN.agravos`.

    Attrs:
        disease (str): The disease to be downloaded.
    """

    SINAN.download_parquets(disease, years, data_path=PYSUS_DATA_PATH)

    logger.info(f"All years for {disease} downloaded at {PYSUS_DATA_PATH}")


def metadata_df(disease: str) -> pd.DataFrame:
    """
    Returns a DataFrame containing metadata for a SINAN disease.
    """
    try:
        return SINAN.metadata_df(disease)
    except Exception:
        logger.error(f"Metadata not available for {disease}")
