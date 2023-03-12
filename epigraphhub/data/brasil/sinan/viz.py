import pandas as pd
from loguru import logger
from pysus import SINAN

from epigraphhub.connection import get_engine
from epigraphhub.data._config import SINAN_LOG_PATH
from epigraphhub.settings import env

from . import normalize_str

logger.add(SINAN_LOG_PATH, retention="7 days")

engine = get_engine(credential_name=env.db.default_credential)


def parquet(disease: str, year: str|int) -> pd.DataFrame:
    """
    Convert the parquet files into a pandas DataFrame.

    Parameters
    ----------
        fname (str)            : Path of the parquet dir.
        clean_after_read (bool): If set to True, will delete the data after
                                 returning the DataFrame.
    Returns
    -------
        df (DataFrame)         : A Pandas DataFrame.
    """

    df = SINAN.parquet_to_df(disease, year)
    df.columns = df.columns.str.lower()

    return df


def table(disease: str) -> pd.DataFrame:
    """
    Connect to EGH SQL server and retrieve the data by disease and year.

    Parameters
    ----------
        disease (str) : The name of the disease according to SINAN.agravos
        year (int)    : Year of the wanted data.
    Returns
    -------
        df (DataFrame): The data requested in a Pandas DataFrame.

    """

    tablename = "sinan_" + normalize_str(disease) + "_m"
    schema = "brasil"

    with engine.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM {schema}.{tablename}", conn)

    return df
