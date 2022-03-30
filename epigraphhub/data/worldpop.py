import json

import pandas as pd
import requests as rq


def json_get(url):
    response = rq.get(url)
    return response.json()


class WorldPop:
    api_root = "https://www.worldpop.org/rest/data"

    def __init__(self):
        self._datasets = None
        self.aliases = [d["alias"] for d in self.datasets["data"]]

    @property
    def datasets(self):
        if self._datasets is None:
            self._datasets = json_get(self.api_root)
        return self._datasets

    def __str__(self):
        df = pd.DataFrame(self.datasets["data"])
        return df.to_markdown()

    def get_dataset_tables(self, alias):
        """
        Generator function that returns dictionarys containing on dataframe and metadata at a time
        Args:
            alias: alias for the dataset of interest
        """
        if alias not in self.aliases:
            raise NameError(f"'{alias}' is invalid dataset alias")

        alias_url = self.api_root + f"/{alias}"
        content = json_get(alias_url)
        for ds in content["data"]:
            table = json_get(alias_url + f"/{ds['alias']}")["data"]
            df = pd.DataFrame(
                table,
            )
            yield {"alias": ds["alias"], "name": ds["name"], "data": df}
