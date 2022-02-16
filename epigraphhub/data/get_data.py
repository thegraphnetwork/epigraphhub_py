"""
The functions in this module allow the user to get each of the datasets made available
by FOPH - Federal Office of Public Health for each canton of Switzerland.

In the future will be datasets for other countries available.   

The function get_georegion_data filter the datasets for a list of 
selected geo regions (in the Switzerland case: cantons).

The function get_cluster_data returns a table where each column is related
with a curve and a canton (e.g. the daily number of cases and the daily number of hospitalizations). This function is handy to apply forecast models in 
these data. 

"""


import pandas as pd
from sqlalchemy import create_engine

engine_public = create_engine(
    "postgresql://epigraph:epigraph@localhost:5432/epigraphhub"
)
engine_private = create_engine(
    "postgresql://epigraph:epigraph@localhost:5432/privatehub"
)


def get_georegion_data(country, georegion, curve, columns):
    """
    This function provide a dataframe for the curve selected in the param curve and
    the canton selected in the param canton

    params country: string. Country that you want get the data, for now, the unique option is
    'Switzerland'.

    param georegion: array with all the subregions of the country of interest or string 'All'
    to return all the georegions.

    param curve: string. One of the following options are accepted: ['cases', 'casesVaccPersons', 'covidCertificates', 'death',
                                                             'deathVaccPersons', 'hosp', 'hospCapacity', 'hospVaccPersons',
                                                             'intCases', 're', 'test', 'testPcrAntigen', 'virusVariantsWgs']

    param columns: columns of interest
    return dataframe
    """
    # put the country name in lower to avoid errors
    country = country.lower()

    # list with the accepted countrys
    accepted_countrys = ["switzerland"]

    accepted_curves = {
        "switzerland": [
            "cases",
            "casesVaccPersons",
            "covidCertificates",
            "death",
            "deathVaccPersons",
            "hosp",
            "hospcapacity",
            "hospVaccPersons",
            "intCases",
            "re",
            "test",
            "testPcrAntigen",
            "virusVariantsWgs",
        ]
    }

    # return a error if a param is not in the correct format.

    if country not in accepted_countrys:

        raise Exception(f"Error. The only countries accepted are: {accepted_countrys}.")

    if curve not in accepted_curves[country]:

        raise Exception(
            f"Error. The only curves accepted are: {accepted_curves[country]}."
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

    # name of the table according to the country
    table_names = {"switzerland": "foph_"}

    # if the columns are not specified get all the columns
    if columns == None:
        columns = "*"

    # separe the columns by comma
    s_columns = ""
    for i in columns:

        s_columns = s_columns + i + ","

    s_columns = s_columns[:-1]

    # getting the data from the database

    # defining the SQL query to get the data
    if georegion == "All":
        query = f"select {s_columns} from {country}.{table_names[country] + curve}"

    if len(georegion) == 1:
        query = f"select {s_columns}  from {country}.{table_names[country] + curve} where \"geoRegion\" = '{georegion[0]}' ;"

    if len(georegion) > 1 and type(georegion) == list:
        georegion_tuple = tuple(i for i in georegion)
        query = f'select {s_columns} from {country}.{table_names[country] + curve} where "geoRegion" in {georegion_tuple} ;'

    df = pd.read_sql(query, engine_public)

    return df


dict_cols = {
    "cases": ['"geoRegion"', "datum", "entries"],
    "test": ['"geoRegion"', "datum", "entries", "entries_pos"],
    "hosp": ['"geoRegion"', "datum", "entries"],
    "hospcapacity": [
        '"geoRegion"',
        "date",
        '"ICU_Covid19Patients"',
        '"Total_Covid19Patients"',
    ],
    "re": ["geoRegion", "date", "median_R_mean"],
}

date_columns = {
    "cases": "datum",
    "test": "datum",
    "hosp": "datum",
    "hospcapacity": "date",
}

count_columns = {
    "cases": ["entries"],
    "test": ["entries"],
    "hosp": ["entries"],
    "hospcapacity": ["ICU_Covid19Patients", "Total_Covid19Patients"],
}


def get_cluster_data(
    curve,
    georegion,
    country="Switzerland",
    dict_cols=dict_cols,
    date_columns=date_columns,
    count_columns=count_columns,
    vaccine=True,
    smooth=True,
):
    """
    This function provide a dataframe where each columns is associated with the curve
    and georegion selected.

    return dataframe
    """

    country = "Switzerland"

    dict_cols = {
        "cases": ['"geoRegion"', "datum", "entries"],
        "test": ['"geoRegion"', "datum", "entries", "entries_pos"],
        "hosp": ['"geoRegion"', "datum", "entries"],
        "hospcapacity": [
            '"geoRegion"',
            "date",
            '"ICU_Covid19Patients"',
            '"Total_Covid19Patients"',
        ],
        "re": ["geoRegion", "date", "median_R_mean"],
    }

    date_columns = {
        "cases": "datum",
        "test": "datum",
        "hosp": "datum",
        "hospcapacity": "date",
    }

    count_columns = {
        "cases": ["entries"],
        "test": ["entries"],
        "hosp": ["entries"],
        "hospcapacity": ["ICU_Covid19Patients", "Total_Covid19Patients"],
    }

    # The dicts above could be passed as params to possibilite adapte this function for another country/subregion.

    # print(df)
    # dataframe where will the curve for each region

    df_end = pd.DataFrame()

    for i in curve:
        # print(i)
        df = get_georegion_data(country, georegion, i, dict_cols[i])
        df.set_index(date_columns[i], inplace=True)
        df.index = pd.to_datetime(df.index)

        for j in df.geoRegion.unique():

            # print(j)

            for k in count_columns[i]:

                if i == "hospcapacity":

                    names = {
                        "ICU_Covid19Patients": "ICU_patients",
                        "Total_Covid19Patients": "total_hosp",
                    }
                    df_aux = df.loc[df.geoRegion == j].resample("D").mean()

                    df_end[names[k] + "_" + j] = df_aux[k]
                    df_end[f"diff_{names[k]}_{j}"] = df_aux[k].diff(1)
                    df_end[f"diff_2_{names[k]}_{j}"] = df_aux[k].diff(2)

                else:
                    df_aux = df.loc[df.geoRegion == j].resample("D").mean()

                    # print(df_end.index)
                    # print(df_aux.index)

                    df_end[i + "_" + j] = df_aux[k]
                    # print(len(df_aux[k]))
                    # print(len(np.concatenate( ([np.nan], np.diff(df_aux[k],1)))))
                    df_end[f"diff_{i}_{j}"] = df_aux[k].diff(1)
                    df_end[f"diff_2_{i}_{j}"] = df_aux[k].diff(2)

    df_end = df_end.resample("D").mean()

    if vaccine == True:
        ## add the vaccine data for Switzerland made available by Our world in Data
        vac = pd.read_csv(
            "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
        )

        # selecting the switzerland data
        vac = vac.loc[vac.iso_code == "CHE"]
        vac.index = pd.to_datetime(vac.date)

        # selecting only the column with vaccinations per hundred
        vac = vac[["total_vaccinations_per_hundred"]]

        vac = vac.fillna(0)

        if vac.total_vaccinations_per_hundred[-1] == 0:
            vac.total_vaccinations_per_hundred[-1] = vac.total_vaccinations_per_hundred[
                -2
            ]

        df_end["vac_all"] = vac.total_vaccinations_per_hundred

    # filling the NaN values by zero
    df_end = df_end.fillna(0)

    if smooth == True:
        df_end = df_end.rolling(window=7).mean()

        df_end = df_end.dropna()

    return df_end


def get_updated_data(smooth=True):

    """
    Function to get the updated data for Geneva

    param smooth: Boolean. If True, a rolling average is applied

    return: dataframe.
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
