import argparse
from pathlib import Path

import yaml
from sqlalchemy.engine.url import make_url

# note: use dash (-) instead of underscore (_)
DEFAULT_CONFIG_CONTENT = {
    "db": {
        "default-credential": "",
        "credentials": {},
    }
}


def create_file_cli_args() -> None:
    parser = argparse.ArgumentParser(
        description="Create a config file for EpiGraphHub.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--name",
        type=str,
        help="Specify the name for this new configuration.",
        required=True,
    )
    parser.add_argument(
        "--db-uri",
        type=str,
        help=(
            "Specify database connection credential.\n"
            "Format (sqalchemy uri):\n"
            "  <username>:<password>@<host>:<posrt>/<dbname>\n"
            "Example: \n"
            "  --db-uri 'epiuser:epipassword@epihost:5432/epidb'"
        ),
        required=False,
    )
    parser.add_argument(
        "--db-default-credential",
        type=str,
        help=("Specify the default database credential."),
        required=False,
    )
    return parser.parse_args()


def create_file() -> None:
    """Create a config file for epigraphhub_py."""

    args = create_file_cli_args()

    content = read()

    if args.db_default_credential:
        content["db"]["default-credential"] = args.db_default_credential

    if args.db_uri:
        url = make_url("postgres://" + args.db_uri)

        content["db"]["credentials"][args.name] = {
            "host": url.host,
            "port": url.port,
            "dbname": url.database,
            "username": url.username,
            "password": url.password,
        }

        if not content["db"]["default-credential"]:
            content["db"]["default-credential"] = args.name

    config_dir = Path.home() / Path(".config")
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / Path("epigraphhub.yaml")

    with open(config_path, "w") as f:
        f.write(yaml.dump(content))


def read() -> dict:
    config_path = Path.home() / Path(".config/epigraphhub.yaml")

    if not config_path.exists():
        return DEFAULT_CONFIG_CONTENT

    with open(config_path) as f:
        return yaml.safe_load(f.read())
