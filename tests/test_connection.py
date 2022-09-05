import pytest

from epigraphhub.connection import Tunnel, get_engine

#
# def test_tunnel():
#     t = Tunnel()
#     t.open_tunnel("epigraph", "")
#     assert t.__repr__().startswith("Port")
#
#
# @pytest.mark.parametrize(("credential_name",), [("epigraphhub",), ("sandbox",)])
# def test_engine(credential_name):
#     e = get_engine(credential_name=credential_name)
#     e.connect()
