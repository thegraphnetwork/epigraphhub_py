import geopandas as gpd

from epigraphhub.connection import get_engine


def upload_geo_file(fname, table_name, schema, db):
    """
    Uploads a georeferenced file to the epigraphhub database
    Args:
        fname:
        table_name:
        schema:
        db:
    """
    gdf = gpd.read_file(fname)
    eng = get_engine(db=db)
    gdf.to_postgis(table_name, con=eng, schema=schema, if_exists="replace", index=False)

    return 