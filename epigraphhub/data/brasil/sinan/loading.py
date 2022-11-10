import os
from pathlib import PosixPath

from loguru import logger
from pangres import upsert
from pysus.online_data import parquets_to_dataframe as to_df

from epigraphhub.connection import get_engine
from epigraphhub.data._config import SINAN_LOG_PATH
from epigraphhub.settings import env

logger.add(SINAN_LOG_PATH, retention="7 days")

engine = get_engine(credential_name=env.db.default_credential)


def upload(parquets: list[PosixPath]):

    if any(parquets) and isinstance(parquets, list):
        for parquet in parquets:
            if any(os.listdir(parquet)):

                df = to_df(str(parquet), clean_after_read=True)
                df.columns = df.columns.str.lower()
                df.index.name = "index"

                table_i = str(parquet.split("/")[-1]).split(".parquet")[0]
                st, yr = table_i[:-4].lower(), table_i[-2:]
                table = "".join([st, yr])
                schema = "brasil"

                with engine.connect() as conn:
                    try:

                        upsert(
                            con=conn,
                            df=df,
                            table_name=table,
                            schema=schema,
                            if_row_exists="update",
                            chunksize=1000,
                            add_new_columns=True,
                            create_table=True,
                        )

                        logger.info(f"Table {table} updated")

                    except Exception as e:
                        logger.error(f"Not able to upsert {table} \n{e}")

    else:
        raise Exception(f"Bad format. Expected {type(list)}, received {type(parquets)}")
