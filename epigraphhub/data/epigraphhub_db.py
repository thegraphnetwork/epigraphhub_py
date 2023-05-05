"""
The functions in this module allow the user to get the datasets stored
in the epigraphhub database.
"""
from typing import Union

import pandas as pd
from sqlalchemy import create_engine

from epigraphhub.settings import env

with env.db.credentials[env.db.default_credential] as credential:
    engine_public = create_engine(
        f"postgresql://{credential.username}:"
        f"{credential.password}@{credential.host}:{credential.port}/"
        f"{credential.dbname}"
    )


def get_agg_data(
    schema: str, table_name: str, columns: list, method: str, ini_date: str
) -> pd.DataFrame:
    """
    This function provides an aggregate DataFrame for the table selected
    in the param table_name. The columns should be a list with three
    values. The first should be a date column, the second a column that
    will be used for the aggregation (e.g. regions name), and the third
    the column that will be used to compute the result of the
    aggregation.

    Parameters
    ----------
    schema : str
        The schema in the epigraphhub database.
    table_name : str
        The name of the table.
    columns : list
        The list of columns from the table that will be used in the
        aggregation. The first column should be a date column, the
        second should be a column with the regions name that we want to
        aggregate (e.g. regions name), and the third will be used to
        compute the result of aggregation.
    method : str
        The method name to be applied in the aggregation, the possible
        options are: 'COUNT', 'SUM',  and 'AVG'.
    ini_date : str
        Initial data to start the aggregation.

    Returns
    -------
    pd.DataFrame
        The return is a pandas DataFrame.
    """

    table_name = table_name.lower()
    method = method.upper()

    query = (
        f"SELECT {columns[0]}, {columns[1]}, {method}({columns[2]}) "
        f"FROM {schema}.{table_name} WHERE {columns[0]} > '{ini_date}' "
        f"GROUP BY ({columns[0]}, {columns[1]})"
    )

    df = pd.read_sql(query, engine_public)
    df.set_index(columns[0], inplace=True)
    df.index = pd.to_datetime(df.index)

    return df


def get_data_by_location(
    schema: str,
    table_name: str,
    loc: Union["list[str]", str],
    columns: "list[str]",
    loc_column: str,
) -> pd.DataFrame:
    """
    This function provides a DataFrame for the table selected in the param
    table_name and the chosen regions in the param georegion.

    Parameters
    ----------
    schema : str
        The schema where the data that you want to get is saved.
    table_name : str
        Name of the table that you want to get the data.
    loc : Union[list[str], str]
        This list contains all the locations of interest or the string 'All' to
        return all the regions.
    columns : list[str], None
        Columns that you want to select from the table table_name. If None all
        the columns will be returned.
    loc_column : str
        Name of the column to filter by location name.

    Returns
    -------
    pd.DataFrame
        _description_

    Raises
    ------
    Exception
        _description_
    Exception
        _description_
    """

    schema = schema.lower()
    table_name = table_name.lower()

    if type(loc) != list and loc != "All":
        raise Exception(
            """Error. The georegion param should be a list or the string
            'All' to return all the georegions."""
        )

    if type(columns) != list and columns != None:
        raise Exception(
            "Error. The columns param should be a list or None. If None all the columns will be returned."
        )

    if columns == None:
        s_columns = "*"

    else:
        # separe the columns by comma to apply in the sql query
        s_columns = ""
        for i in columns:
            s_columns = s_columns + i + ","

        s_columns = s_columns[:-1]

    if loc == "All":
        query = f"select {s_columns} from {schema}.{table_name}"

    if len(loc) == 1:
        query = f"select {s_columns} from {schema}.{table_name} where {loc_column} = '{loc[0]}' ;"

    if len(loc) > 1 and loc != "All":
        loc_tuple = tuple(i for i in loc)
        query = f'select {s_columns} from {schema}.{table_name} where "{loc_column}" in {loc_tuple} ;'

    df = pd.read_sql(query, engine_public)

    return df
