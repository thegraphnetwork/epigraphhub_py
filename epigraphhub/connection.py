from typing import Optional

import atexit
import warnings

from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder

from epigraphhub.settings import env


class Tunnel:
    def __init__(self, host: str):
        self.host: str = host
        self.server = None
        atexit.register(self.close_tunnel)

    def open_tunnel(self, user: str = "epigraph", ssh_key_passphrase=""):
        """
        Opens a tunnel to EpigraphHub database

        Parameters
        ----------
        user : str
            User to be used for the connection.
        ssh_key_passphrase: str
            Your SSH key passphrase.
        """
        if self.host == env.db.host:
            return warnings.warn(
                "Tunnel is not necessary because remote and local "
                "address is the same."
            )

        self.server = SSHTunnelForwarder(
            self.host,
            ssh_username=env.db.username,
            ssh_password=env.db.password,
            ssh_private_key_password=ssh_key_passphrase,
            remote_bind_address=(env.db.host, env.db.password),
        )

        self.server.start()

    def __repr__(self):
        if self.server is None:
            return "Tunnel is closed."
        else:
            return f"Port {self.server.local_bind_port} is being forwarded to {self.host}:5432"

    def close_tunnel(self):
        if not self.server is None:
            print("Closing Tunnel...")
            self.server.stop()
            self.server = None


def get_engine(credential_name: str, db: Optional[str] = None):
    """
    Returns an engine connected to the Epigraphhub database
    """
    with env.db.credentials[credential_name] as credential:
        db = db or credential.dbname
        uri = (
            f"postgresql://{credential.username}:"
            f"{credential.password}@"
            f"{credential.host}:{credential.port}/"
            f"{db}"
        )
        return create_engine(uri)
