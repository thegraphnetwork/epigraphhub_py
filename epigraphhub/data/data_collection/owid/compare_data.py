"""
Last change on 2022/09/22
Comparing Our World in Data (OWID) COVID data consists in
a step before pushing it to the database. Is responsible for retrieving
the table size in both CSV and in the SQL Database.
@see epigraphhub.connection

Methods
-------

database_size(remote):
    If remote, creates the ssh connection with the SQL server. Then returns
    the total count of the OWID table.

csv_size():
    Looks for the OWID CSV file and returns its total rows count.
"""
import os
import shlex as sx
import subprocess

from loguru import logger

from epigraphhub.connection import get_engine
from epigraphhub.data.data_collection.config import (
    OWID_CSV_PATH,
    OWID_FILENAME,
    OWID_HOST,
    OWID_LOG_PATH,
)
from epigraphhub.settings import env

logger.add(OWID_LOG_PATH, retention="7 days")


def database_size(remote=True):
    """
    Method responsible for connecting in the SQL database and return
    the total count of the OWID table.

    Args:
        remote (bool)         : If the SQL container is not locally configured,
                                creates a ssh tunnel with the Database.

    Raises:
        Exception (Exception) : Connection with the Database could not be
                                stablished.
    """
    if remote:
        proc = subprocess.Popen(
            sx.split(f"ssh -f epigraph@{OWID_HOST} -L 5432:localhost:5432 -NC")
        )
    try:
        engine = get_engine(env.db.default_credential)
        with engine.connect().execution_options(autocommit=True) as conn:
            curr = conn.execute("SELECT COUNT(*) FROM owid_covid")
            for count in curr:
                return int(count[0])
    except Exception as e:
        logger.error(f"Could not access OWID table\n{e}")
        raise (e)
    finally:
        if remote:
            proc.kill()


def csv_size():
    """
    Method responsible for connecting in the SQL database and return
    the total count of the OWID table.

    Args:
        remote (bool)         : If the SQL container is not locally configured,
                                creates a ssh tunnel with the Database.

    Raises:
        Exception (Exception) : Connection with the Database could not be
                                stablished.
    """
    try:
        raw_shape = subprocess.Popen(
            f"wc -l {os.path.join(OWID_CSV_PATH, OWID_FILENAME)}",
            shell=True,
            stdout=subprocess.PIPE,
        ).stdout
        clean = str(raw_shape.read()).split("'")
        shape = clean[1].split(" ")[0]
        return int(shape) - 1
    except Exception as e:
        logger.error(f"Could not reach OWID csv\n{e}")
