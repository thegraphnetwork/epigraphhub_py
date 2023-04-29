import geopandas as gpd
import rioxarray

from epigraphhub.connection import get_engine


def upload_geo_file(
    fname: str, table_name: str, schema: str, credential_name: str
) -> None:
    """
    Uploads a georeferenced file to the epigraphhub database.

    Parameters
    ----------
    fname : str
        Name of the file to upload.
    table_name : str
        Table name where the data is saved.
    schema : str
        Schema name where the data is saved.
    credential_name : str
        The credential name to make the connection to the database
    """

    gdf = gpd.read_file(fname)
    eng = get_engine(credential_name=credential_name)
    gdf.to_postgis(table_name, con=eng, schema=schema, if_exists="replace", index=False)


def upload_geotiff(geotiff_file_name, table_name, schema, credential_name):
    """
    Upload Geotiff file to database.
    """
    eng = get_engine(credential_name=credential_name)
    xds = rioxarray.open_rasterio(geotiff_file_name)
