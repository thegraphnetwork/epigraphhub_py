import pytest
import geopandas as gpd
import rioxarray as rx
import os
import wget
from epigraphhub.data.upload import upload_geo_file, upload_geotiff
from epigraphhub.connection import Tunnel
import getpass

def test_upload_geo_file():
    url = "http://download.geofabrik.de/antarctica-latest-free.shp.zip"
    # phrase = getpass.getpass("enter your ssh passphrase")
    # T= Tunnel()`
    # T.open_tunnel("epigraph", phrase)`
    temp_data = gpd.read_file(url)
    temp_data.to_file('test_geopackage.gpkg', driver='GPKG')
    upload_geo_file('test_geopackage.gpkg','antarctica',schema='public',db='sandbox')
    os.unlink('test_geopackage.gpkg')

def test_upload_geotiff():
    url = "https://data.worldpop.org/GIS/AgeSex_structures/school_age_population/v1/2020/LSO/LSO_SAP_1km_2020/LSO_M_PRIMARY_2020_1km.tif"
    wget.download(url, out='temp.tif')
    rx.open_rasterio('temp.tif')

