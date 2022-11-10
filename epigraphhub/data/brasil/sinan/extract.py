from pysus.online_data import SINAN
from pathlib import Path, PosixPath
from loguru import logger

from epigraphhub.data._config import SINAN_LOG_PATH

logger.add(SINAN_LOG_PATH, retention="7 days")

aggravates = SINAN.agravos

def download(
    aggravate: str,
    data_dir: str = "/tmp",
) -> list[PosixPath]:
    """
    Download all parquets available for an aggravate,
    according to `SINAN.agravos`.

    Attrs:
        aggravate (str): The aggravate to be downloaded.
        data_dir (str) : The output directory were files will be downloaded.
                         A directory with the aggravate code will be created.

    Returns:
        parquets_paths_list list(PosixPath) : A list with all parquets dirs.
    """
    data_dir = Path(data_dir) / "pysus" / aggravates[aggravate]
    parquets_paths_list = SINAN.download_all_years_in_chunks(aggravate, data_dir)

    logger.info(f"All years for {aggravate} downloaded at {data_dir}")
    return parquets_paths_list
