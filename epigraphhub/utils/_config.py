import argparse
from pathlib import Path

import yaml


def create_file_cli_args() -> None:
    parser = argparse.ArgumentParser(
        description="Create a config file for EpiGraphHub.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--db-host", type=str, help="DB Host", required=False)
    parser.add_argument("--db-port", type=str, help="DB Port", required=False)
    parser.add_argument(
        "--db-credential",
        type=str,
        action="append",
        help=(
            "Specify a credential, it can be used many time\n"
            "Format:\n"
            "  <credential_name>:<dbname>/<username>/<password>\n"
            "Example: \n"
            "  --db-credential 'epigraph:epidb/epiuser/epipassword'"
        ),
        required=False,
    )
    return parser.parse_args()


def create_file() -> None:
    """Create a config file for epigraphhub_py."""

    args = create_file_cli_args()

    default_content = {
        "db": {
            "host": "localhost",
            "port": "5432",
            "credentials": {
                "postgres": {
                    "dbname": "postgres",
                    "username": "postgres",
                    "password": "postgres",
                },
                "public": {
                    "dbname": "epigraph",
                    "username": "epigraph",
                    "password": "epigraph",
                },
                "private": {
                    "dbname": "epigraph_private",
                    "username": "epigraph_private",
                    "password": "epigraph",
                },
                "sandbox": {
                    "dbname": "epigraph_sandbox",
                    "username": "epigraph_sandbox",
                    "password": "epigraph",
                },
            },
        }
    }

    if args.db_host:
        default_content["db"]["host"] = args.db_host

    if args.db_port:
        default_content["db"]["port"] = args.db_port

    for db_credential in args.db_credential or []:
        if db_credential.count(":") != 1 and db_credential.count(":") != 2:
            raise Exception(
                "Credential is not in the correct format: "
                "<credential_name>:<dbname>/<username>/<password>"
            )
        credential_name, db_value = db_credential.split(":")
        db_name, db_user, db_pass = db_value.split("/")

        default_content["db"]["credentials"][credential_name] = {
            "dbname": db_name,
            "username": db_user,
            "password": db_pass,
        }

    config_path = Path.home() / Path(".config/epigraphhub.yaml")
    with open(config_path, "w") as f:
        f.write(yaml.dump(default_content))


def read() -> dict:
    config_path = Path.home() / Path(".config/epigraphhub.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f.read())
