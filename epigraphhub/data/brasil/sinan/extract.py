import pandas as pd
from loguru import logger
from pysus.online_data import SINAN

from epigraphhub.data._config import PYSUS_DATA_PATH, SINAN_LOG_PATH

logger.add(SINAN_LOG_PATH, retention="7 days")


def download(disease: str, years: list) -> list:
    """
    Download all parquets available for a disease,
    according to `SINAN.agravos`.

    Parameters
    ----------
    disease : str
        The disease to be downloaded.
    years : list
        The years to be downloaded.

    Returns
    -------
    list
        A list with full paths of parquet dirs to upload into db
    """

    parquets_dirs = SINAN.download(
        disease=disease, years=years, data_path=PYSUS_DATA_PATH
    )

    logger.info(f"Disease {disease} for years {years} downloaded at {PYSUS_DATA_PATH}")

    return parquets_dirs


def metadata_df(disease: str) -> pd.DataFrame:
    """
    Returns a DataFrame containing metadata for a SINAN disease.

    Parameters
    ----------
    disease : str
        A SINAN disease.

    Returns
    -------
    pd.DataFrame
    """
    try:
        return SINAN.metadata_df(disease)
    except Exception:
        logger.error(f"Metadata not available for {disease}")
