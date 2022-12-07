import pandas as pd
from loguru import logger
from pysus.online_data import SINAN
from pysus.online_data import parquets_to_dataframe as to_df

from epigraphhub.connection import get_engine
from epigraphhub.data._config import SINAN_LOG_PATH
from epigraphhub.settings import env

logger.add(SINAN_LOG_PATH, retention="7 days")

engine = get_engine(credential_name=env.db.default_credential)
aggrs = SINAN.agravos


def parquet(ppath: str, clean_after_read=False) -> pd.DataFrame:
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

    df = to_df(str(ppath), clean_after_read)
    logger.info("Parquet files converted to dataFrame")
    df.columns = df.columns.str.lower()

    return df


def table(disease: str, year: int) -> pd.DataFrame:
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

    year = str(year)[-2:].zfill(2)
    disease = SINAN.check_case(disease)
    dis_code = aggrs[disease].lower()
    tablename = f"{dis_code}{year}"

    with engine.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM brasil.{tablename}", conn)

    return df
