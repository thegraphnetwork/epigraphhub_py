from typing import Any, Dict, List, Optional, Set, Tuple, Union

from collections import defaultdict

import pandas as pd
import requests as rq


def json_get(url: str, pars: dict[str, Any] = {}) -> dict[str, Any]:
    if pars:
        response = rq.get(url, params=pars)
    else:
        response = rq.get(url)
    return response.json()


class WorldPop:
    """
    Class to Consume the WorldPop REST API
    """

    api_root: str = "https://www.worldpop.org/rest/data"

    def __init__(self):
        self._datasets: dict[str, Any] = {}
        self._aliases: list[str] = [d["alias"] for d in self.datasets["data"]]
        self._levels: dict[str, str] = defaultdict(lambda: [])

    @property
    def datasets(self):
        if not self._datasets:
            self._datasets = json_get(self.api_root)
        return self._datasets

    def __repr__(self) -> str:
        df = pd.DataFrame(self.datasets["data"])
        return df.to_markdown(index=False)

    def get_dataset_tables(self, alias: str) -> dict[str, Union[str, pd.DataFrame]]:
        """
        Generator function that returns dictionaries containing on DataFrame and
        metadata at a time.

        Parameters
        ----------
        alias : str
            Alias for the dataset of interest.
        """
        if alias not in self._aliases:
            raise NameError(f"'{alias}' is invalid dataset alias")

        alias_url = self.api_root + f"/{alias}"
        content = json_get(alias_url)
        for ds in content["data"]:
            self._levels[alias].append(ds["alias"])
            table = json_get(alias_url + f"/{ds['alias']}")["data"]
            df = pd.DataFrame(
                table,
            )
            yield {"alias": ds["alias"], "name": ds["name"], "data": df}

    def get_data_by_country(
        self, alias: str, level: str, ISO3_code: str = "BRA"
    ) -> dict[str, Any]:
        """
        Returns Country specific dataset metadata from given `alias` and `level`.

        Parameters
        ----------
        alias : str
            Top level category of data.
        level : str
            2nd level category of data.
        ISO3_code: str
            3-letter country code.

        Returns:
        --------
        dict
        """
        if alias not in self._aliases:
            raise NameError(f"'{alias}' is invalid dataset alias")
        content = json_get(self.api_root + f"/{alias}/{level}", {"iso3": ISO3_code})
        return content
