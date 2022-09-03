from dataclasses import dataclass, field
from typing import List

from epigraphhub.utils._config import read

__all__ = ["config"]

config_data = read()


@dataclass
class DBCredential:
    dbname: str = "placeholder"
    username: str = "placeholder"
    password: str = "placeholder"


@dataclass
class DBConfig:
    host: str = "localhost"
    port: str = "5432"
    credentials: dict[str, DBCredential] = field(default_factory=dict)


@dataclass
class Config:
    db: DBConfig = DBConfig()


db_data = config_data["db"]
db_credentials = db_data["credentials"]

print(db_credentials.values())

for credential_data in list(db_credentials.values()):
    print(credential_data)

env = Config(
    db=DBConfig(
        host=db_data["host"],
        port=db_data["port"],
        credentials={
            name: DBCredential(
                dbname=credential_data["dbname"],
                username=credential_data["username"],
                password=credential_data["password"]
            )
            for name, credential_data in db_credentials.items()
        }
    )
)
