from typing import List

from dataclasses import dataclass, field

from epigraphhub.utils._config import read

__all__ = ["config"]

config_data = read()


class ConfigContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@dataclass
class DBCredential(ConfigContext):
    host: str = "localhost"
    port: str = "5432"
    dbname: str = "placeholder"
    username: str = "placeholder"
    password: str = "placeholder"


@dataclass
class DBConfig(ConfigContext):
    default_credential: str = "epigraphhub"
    credentials: dict[str, DBCredential] = field(default_factory=dict)


@dataclass
class Config(ConfigContext):
    db: DBConfig = DBConfig()


db_data = config_data["db"]
db_credentials = db_data["credentials"]

for credential_data in list(db_credentials.values()):
    print(credential_data)

env = Config(
    db=DBConfig(
        default_credential=db_data["default-credential"] or "epigraphhub",
        credentials={
            name: DBCredential(
                host=credential_data["host"],
                port=credential_data["port"],
                dbname=credential_data["dbname"],
                username=credential_data["username"],
                password=credential_data["password"],
            )
            for name, credential_data in db_credentials.items()
        },
    )
)
