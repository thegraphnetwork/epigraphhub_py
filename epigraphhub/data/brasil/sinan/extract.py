from typing import Optional, Union

import shutil
import time
from collections import ChainMap, defaultdict
from functools import lru_cache
from pathlib import Path, PosixPath

from loguru import logger
from pysus.online_data import SINAN

from epigraphhub.data._config import SINAN_LOG_PATH

st = time.time()
logger.add(SINAN_LOG_PATH, retention="7 days")
aggrs = SINAN.list_diseases()


def download(
    aggravate: str,
    year: Union[str, int],
    data_dir: Optional[str] = "/tmp",
):
    try:
        return _download_from_sinan(aggravate, year, data_dir)

    except Exception as e:
        logger.error(e)


def years(aggravate) -> list:

    aggravate = aggravate.title()

    if aggravate in aggrs:
        return _fetch_years_by_aggravate()[aggravate]

    else:
        raise Exception(logger.error(f"{aggravate} not found in {aggrs}"))


def aggravates(year) -> list:

    year = str(year)[-2:]
    aggravates_by_years = _fetch_aggravates_by_year()

    try:
        return aggravates_by_years[year]

    except:
        raise Exception(f"Year {year} not found in SINAN database.")


@lru_cache
def _fetch_years_by_aggravate() -> dict:

    years_by_aggravate = list()

    for aggravate in aggrs:
        years = SINAN.get_available_years(aggravate)
        dis_years = {aggravate: [year.split(".dbc")[0][-2:] for year in years]}
        years_by_aggravate.append(dis_years)

    years_by_aggravate = dict(ChainMap(*years_by_aggravate))

    return years_by_aggravate


# Caching data
i_data = _fetch_years_by_aggravate()


@lru_cache
def _fetch_aggravates_by_year() -> dict:

    years_by_aggravate = _fetch_years_by_aggravate()
    aggravates_by_year = defaultdict(list)

    for aggravate, years in years_by_aggravate.items():
        for year in years:
            aggravates_by_year[year] += [aggravate]

    return dict(aggravates_by_year)


def _download_from_sinan(aggravate: str, year: int, data_dir: str) -> PosixPath:

    aggravate = aggravate.title()
    cod_aggr = SINAN.agravos.get(aggravate)

    year = str(year)[-2:]

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    fname = Path(f"{cod_aggr.upper()}BR{str(year)}.parquet/")
    data_path = Path(data_dir / fname)

    if not data_path.exists():
        if aggravate in aggrs:

            available_years = _fetch_years_by_aggravate()[aggravate]

            if str(year) in available_years:

                elapsed_time = time.time() - st
                fname = Path(SINAN.download(int(year), aggravate, return_fname=True))
                elapsed_time = time.time() - st
                ftime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                logger.info(f"{fname} downloaded at {data_dir} in {ftime}.")

                # Move file to `data_dir`
                shutil.move(fname, data_path)

                return data_path.absolute()

            else:
                logger.error(
                    f"\n{year} not available for {aggravate}."
                    + f"\nAvailable years: {available_years}"
                )

        else:
            logger.error(
                f"\n{aggravate} not found in SINAN aggravates."
                + f"\nAvailable aggravates: {aggrs}"
            )

    else:
        return data_path.absolute()
