import os
from pathlib import Path

import pandas as pd
from loguru import logger
from pysus.online_data import SINAN

from epigraphhub.data._config import PYSUS_DATA_PATH, SINAN_LOG_PATH
from epigraphhub.data.brasil.sinan import DISEASES

logger.add(SINAN_LOG_PATH, retention="7 days")


def download(disease: str) -> None:
    """
    Download all parquets available for a disease,
    according to `SINAN.agravos`.

    Attrs:
        disease (str): The disease to be downloaded.
    """

    SINAN.download_all_years_in_chunks(disease, data_dir=PYSUS_DATA_PATH)

    logger.info(f"All years for {disease} downloaded at {PYSUS_DATA_PATH}")


def metadata_df(disease: str) -> pd.DataFrame:
    """
    Reads metadata sheets located at `./metadata` as DataFrames.
    """
    code = DISEASES[disease]
    metadata_file = f"{code}.xlsx"
    metadata_dir = Path(__file__).parent / "metadata"

    metadata_avail = [
        code_m[:4] for code_m in os.listdir(metadata_dir) if code_m.endswith(".xlsx")
    ]

    if code in metadata_avail:
        return pd.read_excel(metadata_dir / metadata_file)

    else:
        logger.error(f"Metadata not available for {disease}")
