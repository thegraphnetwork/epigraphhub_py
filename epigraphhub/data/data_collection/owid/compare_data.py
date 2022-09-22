import os
import subprocess
import shlex as sx
from epigraphhub.data.data_collection.config import (
    OWID_HOST,
    OWID_CSV_PATH,
    OWID_FILENAME,
    OWID_LOG_PATH,
)
from epigraphhub.connection import get_engine
from epigraphhub.settings import env
from loguru import logger

logger.add(OWID_LOG_PATH, retention="7 days")


def database_size(remote=True):
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
