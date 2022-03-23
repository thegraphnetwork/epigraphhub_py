import numpy as np
import pandas as pd
import wbgapi as wb


def get_pop_data(country, time="all", fx_et="5Y"):

    """
    Function to get the population data stratified by age and sex from
    the world bank data

    :params country: string. ISO-CODE of the country of interest.

    :params time: Interval of  years. If filled `time = 'all'`, the function will
                  return all the data available. You can also specify a range
                  of years. For example, if you want to get the data for the
                  period between the years 2010 and 2020, you can fill this
                  parameter with `time = range(2010,2021)`.

    :params fx_et: string. If fx_et == '5Y', it will be returned the
                          population by 5-year age groups.
                          If fx_et == 'IN', it will be return the
                          population divided in 3 age groups.
                          If fx_et == 'TOTL', it will be returned the
                          total population without consider the age groups.

    :returns: DataFrame.
    """

    fx_et = fx_et.upper()

    if fx_et == "TOTL":

        ind = [
            "SP.POP.TOTL.FE.IN",
            "SP.POP.TOTL.MA.IN",
        ]

    else:
        ind = []

        for i in wb.series.list():

            if (i["id"].startswith("SP.POP")) and (i["id"].endswith(fx_et)):

                ind.append(i["id"])

    df = wb.data.DataFrame(series=ind, economy=country, time=time)

    df = df.T

    new_cols = []

    for i in df.columns:

        new_cols.append(((i[3:-3]).lower()).replace(".", "_"))

    df.columns = new_cols

    df.index = df.index.str.strip(to_strip="YR")

    df["frequency"] = ["yearly"] * len(df)

    df["country"] = [country] * len(df)

    df.index = pd.to_datetime(df.index, format="%Y")
    return df


def search_in_database(keyword):
    """
    Returns a dataframe with the database matched.

    :params keyword: string. Full name or keyword to search in the database
                            names. If the string 'all' is used in the parameter keyword all the
                            databases names are returned.

    :returns: DataFrame.

    """

    results = []

    if keyword.lower() == "all":
        for source in wb.source.list():
            results.append(source)

    else:

        keyword = keyword.replace(" ", "").lower()

        for source in wb.source.list():

            if keyword in source["name"].replace(" ", "").lower():

                results.append(source)

    return pd.DataFrame.from_records(results)


def search_in_indicators(keyword, db=2):

    """
    Returns a dataframe with the indicators matched by partial name.

    :params keyword: string. keyword to search in the indicators name.
    :params db:int. Number associated with the database whose you want to get the list
                    of indicators. You can discover this number in the function 'search_in
                    _database'. By default the indicators are search over the World Development
                    Indicators database (db = 2).

    :returns: DataFrame

    """

    results = []

    for indicator in wb.series.list(q=keyword, db=db):
        results.append(indicator)

    return pd.DataFrame.from_records(results)


def get_worldbank_data(ind, country, db=2, time="all", columns=None):

    """
    This function get a list of indicators according to some country from the world data bank
    and return this series in a dataframe.

    :params ind: List of strings. List with the indicators whose data you want to get.

    :params country: List of strings|string. List with the ISO-CODE for the countries whose
                     data you want to get.

    :params time: Interval of  years. If filled `time = 'all'`, the function will
                  return all the data available. You can also specify a range of years.
                  For example, if you want to get the data for the period between the
                  years 2010 and 2020, you can fill this parameter with
                  `time = range(2010,2021)`.

    :params columns: List of strings. List with names to rename the columns instead of use the
                    names in the `ind` list.

    Important: The lists `ind` and `columns` must have the same lenght.

    :returns: DataFrame

    """

    if columns != None:

        if len(columns) == len(ind):

            rename_columns = {}

            for i in np.arange(0, len(ind)):
                rename_columns[ind[i]] = columns[i]

    else:

        rename_columns = {}

    if len(ind) == 1:

        df = wb.data.DataFrame(
            series=ind, economy=country, db=db, index=["time"], labels=True, time=time
        )

        df = df.reset_index()

        del df["time"]

        df["Time"] = df["Time"].astype("str")

    else:
        df = wb.data.DataFrame(
            series=ind,
            economy=country,
            db=db,
            index=["time", "economy"],
            labels=True,
            time=time,
        )

    if len(rename_columns) != 0:

        df = df.rename(columns=rename_columns)

    new_dates = []
    periodicity = []

    for i in df["Time"]:

        if len(i) == 4:

            new_dates.append(f"{i}-01-01")
            periodicity.append("yearly")

        else:

            if i[4] == "M":

                new_dates.append(f"{i[:4]}-{i[-2:]}-01")
                periodicity.append("monthly")

            if i[4] == "Q":

                dict_month = {
                    "Q1": "01-01",
                    "Q2": "04-01",
                    "Q3": "07-01",
                    "Q4": "10-01",
                }

                new_dates.append(f"{i[:4]}-{dict_month[i[-2:]]}")
                periodicity.append("quarterly")

    df["dates"] = new_dates
    df["frequency"] = periodicity

    df.set_index("dates", inplace=True)

    df.index = pd.to_datetime(df.index)

    if len(ind) == 1:

        df_new = pd.DataFrame()

        for i in country:

            df_aux = pd.DataFrame()

            df_aux[ind[0].lower().replace(".", "_")] = df[i]

            df_aux["country"] = [i] * len(df_aux)

            df_aux["frequency"] = df["frequency"]

            df_new = pd.concat([df_new, df_aux])

        df_new = df_new.sort_index()

        df_new = df_new.dropna()

    else:

        del df["Time"]

        df_new = df

        df_new.columns = df_new.columns.str.lower().str.replace(".", "_")

        df_new.sort_index(inplace=True)

        df_new = df_new.dropna()

    return df_new
