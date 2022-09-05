"""
The functions in this module allow the user to get the datasets stored in the
epigraphhub database.

The function get_agg_data aggregate the data according to the values of one column,
and the method of aggregation applied.

The function get_georegion_data filter the datasets for a list of
selected regions (in the Switzerland case: cantons).

The function get_cluster_data is focused on being used to apply the forecast models.
This function returns a table where each column is related to a  different table and
region (e.g. the daily number of cases and the daily number of hospitalizations in
Geneva and Fribourg).
Some parts of the code of this function are focused on the swiss case.
So the function isn't fully general.
"""


import pandas as pd
from sqlalchemy import create_engine

from epigraphhub.settings import env

with env.db.credentials[env.db.default_credential] as credential:
    engine_public = create_engine(
        f"postgresql://{credential.username}:"
        f"{credential.password}@{credential.host}:{credential.port}/"
        f"{credential.dbname}"
    )


def get_agg_data(schema, table_name, columns, method, ini_date):
    """
    This function provides an aggregate data frame for the table selected in the param
    table_name. The columns should be a list with three values. The first should be
    a date column, the second a column that will be used for the aggregation
    (e.g. regions name), and the third the column that will be used to compute the
    result of the aggregation.

    :params schema: string. The country that you want to get the data, for now, the
                    only options are: ['switzerland', 'colombia'].

    :params table_name: string. Name of the table that you want to get the data.

    :params columns: list of strings. Columns from the table that will be used in the
                    aggregation. The first column should be a date column,
                    the second should be a column with the regions name that we want
                    to aggregate (e.g. regions name), and the third will be used
                    to compute the result of aggregation.

    :params method: string. The method name to be applied in the aggregation the
                            possible options are: 'COUNT', 'SUM',  and 'AVG'.

    :params ini_date: string. Initial data to start the aggregation.

    :return: Dataframe
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
