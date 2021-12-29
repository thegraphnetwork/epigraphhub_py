import geopandas as gpd
import pandas as pd


def upload_geo_file(fname):
    gdf = gpd.read_file(fname)
