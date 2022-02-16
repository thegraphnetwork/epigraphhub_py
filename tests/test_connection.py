import pytest

from epigraphhub.connection import Tunnel, get_engine

#
# def test_tunnel():
#     t = Tunnel()
#     t.open_tunnel("epigraph", "")
#     assert t.__repr__().startswith("Port")
#
#
# @pytest.mark.parametrize(("db",), [("epigraphhub",), ("sandbox",)])
# def test_engine(db):
#     e = get_engine(db=db)
#     e.connect()
