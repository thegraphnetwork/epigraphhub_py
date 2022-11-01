import time

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from pysus.online_data import SINAN

from epigraphhub.data._config import SINAN_LOG_PATH

st = time.time()
logger.add(SINAN_LOG_PATH, retention="7 days")
aggrs = SINAN.list_diseases()


def parquet_to_df(fname: str) -> pd.DataFrame:
    """
    Convert the parquet files into a pandas DataFrame.

    Parameters
    ----------
        fname: Name of the parquet files.
    Returns
    -------
        dataframe: pandas.
    """

    df = (
        pq.ParquetDataset(
            f"{fname}/",
            use_legacy_dataset=False,
        )
        .read_pandas()  # columns=COL_NAMES
        .to_pandas()
    )

    # Measure execution time ended
    logger.info("Convert parquet files to dataFrame, decoding...")
    df.columns = df.columns.str.lower()
    df = df.apply(lambda x: x.str.decode("iso-8859-1"))

    return df
