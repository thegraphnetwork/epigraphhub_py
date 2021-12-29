import pytest

from epigraphhub_py.connection import Tunnel


def test_tunnel():
    t = Tunnel()
    t.open_tunnel("epigraph", "")
    assert t.__repr__().startswith("Port")
