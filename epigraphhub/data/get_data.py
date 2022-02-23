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

engine_public = create_engine(
    "postgresql://epigraph:epigraph@localhost:5432/epigraphhub"
)
engine_private = create_engine(
    "postgresql://epigraph:epigraph@localhost:5432/privatehub"
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

    accepted_countrys = ["switzerland", "colombia"]

    if schema not in accepted_countrys:
        raise Exception(f"Error. The only countries accepted are: {accepted_countrys}.")

    query = f"SELECT {columns[0]}, {columns[1]}, {method}({columns[2]}) FROM {schema}.{table_name} WHERE {columns[0]} > '{ini_date}' GROUP BY ({columns[0]}, {columns[1]})"

    df = pd.read_sql(query, engine_public)
    df.set_index(columns[0], inplace=True)
    df.index = pd.to_datetime(df.index)

    return df


def get_georegion_data(schema, table_name, georegion, columns):
    """
    This function provides a data frame for the table selected in the param table_name and
    the chosen regions in the param georegion.

    ":params schema: string. The country that you want to get the data, for now, the only options are:
                            ['switzerland', 'colombia'].

    :params table_name: string. Name of the table that you want to get the data.

    :param sgeoregion: list of strings| string. This list contains all the regions of the country of interest or the string 'All'
                            to return all the regions.

    :params columns: list of strings| None. Columns that you want to select from the table table_name. If None all the columns will be returned.

    :return: Dataframe
    """

    schema = schema.lower()
    table_name = table_name.lower()

    accepted_countrys = ["switzerland"]

    accepted_tables = {
        "switzerland": [
            "foph_cases",
            "foph_casesVaccPersons",
            "foph_covidCertificates",
            "foph_death",
            "foph_deathVaccPersons",
            "foph_hosp",
            "foph_hospcapacity",
            "foph_hospVaccPersons",
            "foph_intCases",
            "foph_re",
            "foph_test",
            "foph_testPcrAntigen",
            "foph_virusVariantsWgs",
        ],
        "colombia": ["casos_positivos_covid"],
    }

    if schema not in accepted_countrys:
        raise Exception(f"Error. The only countries accepted are: {accepted_countrys}.")

    if table_name not in accepted_tables[schema]:
        raise Exception(
            f"Error. The only curves accepted are: {accepted_tables[schema]}."
        )

    if type(georegion) != list and georegion != "All":
        raise Exception(
            """Error. The georegion param should be a list or the string All to 
        return all the georegions."""
        )

    if type(columns) != list and columns != None:
        raise Exception(
            "Error. The columns param should be a list or None. If None all the columns will be returned."
        )

    if columns == None:
        columns = "*"

    # separe the columns by comma to apply in the sql query
    s_columns = ""
    for i in columns:

        s_columns = s_columns + i + ","

    s_columns = s_columns[:-1]

    if georegion == "All":
        query = f"select {s_columns} from {schema}.{table_name}"

    if len(georegion) == 1:
        query = f"select {s_columns}  from {schema}.{table_name} where \"{columns[1][1:-1]}\" = '{georegion[0]}' ;"

    if len(georegion) > 1 and type(georegion) == list:
        georegion_tuple = tuple(i for i in georegion)
        query = f'select {s_columns} from {schema}.{table_name} where "{columns[1][1:-1]}" in {georegion_tuple} ;'

    df = pd.read_sql(query, engine_public)

    return df


dict_cols = {
    "foph_cases": ["datum", '"geoRegion"', "entries"],
    "foph_test": ["datum", '"geoRegion"', "entries", "entries_pos"],
    "foph_hosp": ["datum", '"geoRegion"', "entries"],
    "foph_hospcapacity": [
        "date",
        '"geoRegion"',
        '"ICU_Covid19Patients"',
        '"Total_Covid19Patients"',
    ],
    "foph_re": ["date", "geoRegion", "median_R_mean"],
}

date_columns = {
    "foph_cases": "datum",
    "foph_test": "datum",
    "foph_hosp": "datum",
    "foph_hospcapacity": "date",
}

count_columns = {
    "foph_cases": ["entries"],
    "foph_test": ["entries"],
    "foph_hosp": ["entries"],
    "foph_hospcapacity": ["ICU_Covid19Patients", "Total_Covid19Patients"],
}

columns_name = {"foph_cases": "cases", "foph_test": "test", "foph_hosp": "hosp"}


def get_cluster_data(
    schema,
    table_name,
    georegion,
    dict_cols=dict_cols,
    date_columns=date_columns,
    count_columns=count_columns,
    columns_name=columns_name,
    vaccine=True,
    smooth=True,
):

    """
    This function provides a data frame where each column is associated with a table
    and region selected.

    :params schema: string. The country that you want to get the data, for now, the only \
                    options are: ['switzerland', 'colombia']

    :params table_name: list of strings. In this list should be all the tables that you 
                        want get the data. 

    :params georegion: list of strings. This list contains all the regions of the country 
                        of interest or the string 'All' to return all the regions.

    :params dict_cols: dictionary. In the keys are the table_names and in the values 
                      the columns that you want to use from each table
    
    :params date_columns:dictionary. In the keys are the table_names and in the values
                          the name of the date column of the table to be used as the 
                          index. 

    :params count_columns: dictionary. In the keys are the table_names and in the values
                        the name of the column which values will be used. 
    
    :params columns_name: dictionary. In the keys ate the table_names and in the values
                        the name that will appear in the column associated with each 
                    table in the final data frame that will be returned. 
        
    :params vaccine: boolean. If True the data of total vaccinations per hundred for the
                            country in the schema will be added in the final data frame. 
                            This data is from our world in data. 
                            
    :params smooth: boolean. If True in the end data frame will be applied a moving 
                            average of seven days. 

    :return: Dataframe
    """

    df_end = pd.DataFrame()

    for table in table_name:

        df = get_georegion_data(schema, table, georegion, dict_cols[table])
        df.set_index(date_columns[table], inplace=True)
        df.index = pd.to_datetime(df.index)

        for region in df.geoRegion.unique():

            for count in count_columns[table]:

                if table == "foph_hospcapacity":

                    names = {
                        "ICU_Covid19Patients": "ICU_patients",
                        "Total_Covid19Patients": "total_hosp",
                    }
                    df_aux1 = df.loc[df.geoRegion == region].resample("D").mean()

                    df_aux2 = pd.DataFrame()

                    df_aux2[names[count] + "_" + region] = df_aux1[count]
                    df_aux2[f"diff_{names[count]}_{region}"] = df_aux1[count].diff(1)
                    df_aux2[f"diff_2_{names[count]}_{region}"] = df_aux1[count].diff(2)

                    df_end = pd.concat([df_end, df_aux2], axis=1)

                else:
                    df_aux1 = df.loc[df.geoRegion == region].resample("D").mean()

                    df_aux2 = pd.DataFrame()

                    df_aux2[columns_name[table] + "_" + region] = df_aux1[count]
                    df_aux2[f"diff_{columns_name[table]}_{region}"] = df_aux1[
                        count
                    ].diff(1)
                    df_aux2[f"diff_2_{columns_name[table]}_{region}"] = df_aux1[
                        count
                    ].diff(2)

                    df_end = pd.concat([df_end, df_aux2], axis=1)

    df_end = df_end.resample("D").mean()

    if vaccine == True:
        vac = pd.read_sql_table(
            "owid_covid",
            engine_public,
            schema="public",
            columns=["date", "iso_code", "total_vaccinations_per_hundred"],
        )

        dict_iso_code = {"switzerland": "CHE", "colombia": "COL"}

        vac = vac.loc[vac.iso_code == dict_iso_code[schema]]
        vac.index = pd.to_datetime(vac.date)

        # selecting only the column with vaccinations per hundred
        vac = vac[["total_vaccinations_per_hundred"]]

        vac = vac.fillna(method="ffill")

        df_end["vac_all"] = vac.total_vaccinations_per_hundred

        df_end["vac_all"] = df_end["vac_all"].fillna(method="ffill")

    df_end = df_end.fillna(0)

    if smooth == True:
        df_end = df_end.rolling(window=7).mean()

        df_end = df_end.dropna()

    return df_end


def get_updated_data_swiss(smooth=True):

    """
    Function to get the updated data for Geneva

    :params smooth: Boolean. If True, a rolling average is applied

    :return: Dataframe.
    """

    df = pd.read_sql_table(
        "hug_hosp_data",
        engine_private,
        schema="switzerland",
        columns=["Date_Entry", "Patient_id"],
    )

    df.index = pd.to_datetime(df.Date_Entry)
    df_hosp = df.resample("D").count()
    df_hosp = df_hosp[["Patient_id"]]

    if smooth == True:
        df_hosp = df_hosp[["Patient_id"]].rolling(window=7).mean()
        df_hosp = df_hosp.dropna()

    df_hosp = df_hosp.sort_index()
    df_hosp.rename(columns={"Patient_id": "hosp_GE"}, inplace=True)

    return df_hosp.loc[df_hosp.index >= "2021-09-01"]
