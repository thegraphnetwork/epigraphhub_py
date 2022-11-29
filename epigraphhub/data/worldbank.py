import numpy as np
import pandas as pd
import wbgapi as wb


def get_pop_data(country, time="all", fx_et="5Y"):
    """
    Function to get the population data stratified by age and sex from
    the world bank data.

    Parameters
    ----------
    country : str
        ISO-CODE of the country of interest.
    time : range, str
        Interval of years. If filled `time = 'all'`, the function will
        return all the data available. You can also specify a range of
        years. For example, if you want to get the data for the period
        between the years 2010 and 2020, you can fill this parameter
        with `time = range(2010,2021)`.
    fx_et : str
        If fx_et == '5Y', it will be returned the population by 5-year
        age groups. If fx_et == 'IN', it will be return the population
        divided in 3 age groups. If fx_et == 'TOTL', it will be returned
        the total population without consider the age groups.

    Returns
    -------
    pd.DataFrame
        The return is a pandas DataFrame.
    """

    fx_et = fx_et.upper()

    ind = {
        "5Y": [
            "SP.POP.0004.FE.5Y",
            "SP.POP.0004.MA.5Y",
            "SP.POP.0509.FE.5Y",
            "SP.POP.0509.MA.5Y",
            "SP.POP.1014.FE.5Y",
            "SP.POP.1014.MA.5Y",
            "SP.POP.1519.FE.5Y",
            "SP.POP.1519.MA.5Y",
            "SP.POP.2024.FE.5Y",
            "SP.POP.2024.MA.5Y",
            "SP.POP.2529.FE.5Y",
            "SP.POP.2529.MA.5Y",
            "SP.POP.3034.FE.5Y",
            "SP.POP.3034.MA.5Y",
            "SP.POP.3539.FE.5Y",
            "SP.POP.3539.MA.5Y",
            "SP.POP.4044.FE.5Y",
            "SP.POP.4044.MA.5Y",
            "SP.POP.4549.FE.5Y",
            "SP.POP.4549.MA.5Y",
            "SP.POP.5054.FE.5Y",
            "SP.POP.5054.MA.5Y",
            "SP.POP.5559.FE.5Y",
            "SP.POP.5559.MA.5Y",
            "SP.POP.6064.FE.5Y",
            "SP.POP.6064.MA.5Y",
            "SP.POP.6569.FE.5Y",
            "SP.POP.6569.MA.5Y",
            "SP.POP.7074.FE.5Y",
            "SP.POP.7074.MA.5Y",
            "SP.POP.7579.FE.5Y",
            "SP.POP.7579.MA.5Y",
            "SP.POP.80UP.FE.5Y",
            "SP.POP.80UP.MA.5Y",
        ],
        "TOTL": ["SP.POP.TOTL.FE.IN", "SP.POP.TOTL.MA.IN"],
        "IN": [
            "SP.POP.0014.FE.IN",
            "SP.POP.0014.MA.IN",
            "SP.POP.1564.FE.IN",
            "SP.POP.1564.MA.IN",
            "SP.POP.65UP.FE.IN",
            "SP.POP.65UP.MA.IN",
            "SP.POP.TOTL.FE.IN",
            "SP.POP.TOTL.MA.IN",
        ],
    }

    df = wb.data.DataFrame(series=ind[fx_et], economy=country, db=2, time=time)

    df = df.T

    df.index = pd.to_datetime(df.index, format="YR%Y")

    df.columns.name = ""

    df.columns = ((df.columns.str.lower()).str.replace(".", "_")).str[3:-3]

    df["frequency"] = "yearly"

    df["country"] = country

    return df


def search_in_database(keyword):
    """
    Returns a DataFrame with the database matched.

    Parameters
    ----------
    keyword : str
        Full name or keyword to search in the database names. If the
        string 'all' is used, all the databases names are returned.

    Returns
    -------
    pd.DataFrame
        The return is a pandas DataFrame.
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
    Returns a DataFrame with the indicators matched by partial name.

    Parameters
    ----------
    keyword : str
        Keyword to search in the indicators name. If None, all the
        indicators available will be returned.
    db : int
        Number associated with the database whose you want to get the
        list of indicators. You can discover this number in the function
        'search_in_database'. By default the indicators are search over
        the World Development Indicators database (db = 2).

    Returns
    -------
    pd.DataFrame
        The return is a pandas DataFrame.
    """

    results = []

    for indicator in wb.series.list(q=keyword, db=db):
        results.append(indicator)

    return pd.DataFrame.from_records(results)


def get_worldbank_data(ind, country, db=2, time="all", columns=None):
    """
    This function get a list of indicators according to some country
    from the world data bank and return this series in a DataFrame.

    Parameters
    ----------
    ind : list
        List with the indicators whose data you want to get.
    country : list
        List with the ISO-CODE for the countries whose data you want to
        get.
    time : range, str
        Interval of years. If filled `time = 'all'`, the function will
        return all the data available. You can also specify a range of
        years. For example, if you want to get the data for the period
        between the years 2010 and 2020, you can fill this parameter
        with `time = range(2010,2021)`.
    columns : list
        List with names to rename the columns instead of use the names
        in the `ind` list.

    Important
    ---------
    The lists `ind` and `columns` must have the same lenght.

    Returns
    -------
    pd.DataFrame
        The return is a pandas DataFrame.
    """

    if len(ind) == 1:

        df = wb.data.DataFrame(
            series=ind[0],
            economy=country,
            db=db,
            index=["time"],
            labels=True,
            time=time,
        )

        df = df.reset_index()

        df["time"] = df["time"].astype("str")

        df = pd.melt(
            df,
            id_vars=["time"],
            value_vars=country,
            var_name="country",
            value_name=ind[0],
        )

        n_columns = ["time", "country"] + list(ind)

        df = df[n_columns]

    else:
        df = wb.data.DataFrame(
            series=ind,
            economy=country,
            db=db,
            index=["time", "economy"],
            labels=True,
            time=time,
        )

        df.reset_index(inplace=True)

        df = df.rename(columns={"economy": "country"})

        n_columns = ["time", "country"] + list(ind)

        df = df[n_columns]

    if columns != None:

        if len(columns) == len(ind):

            rename_columns = dict(zip(ind, columns))

            df = df.rename(columns=rename_columns)

            df.columns = df.columns.str.lower().str.replace(".", "_")

        else:
            raise Exception(
                f"Error. The ind and columns list must have the same length."
            )

    else:

        df.columns = df.columns.str.lower().str.replace(".", "_")

    df["format"] = 1
    df.loc[df.time.str.contains("Q"), "format"] = 2
    df.loc[df.time.str.contains("M"), "format"] = 3
    df.loc[df.time.str.contains("YR"), "format"] = 4
    df["date"] = np.nan

    # Convert to datetime with two different format settings
    df.loc[df.format == 1, "date"] = pd.to_datetime(df.loc[df.format == 1, "time"])

    df.loc[df.format == 2, "date"] = pd.to_datetime(df.loc[df.format == 2, "time"])

    df.loc[df.format == 3, "date"] = pd.to_datetime(
        df.loc[df.format == 3, "time"], format="%YM%m"
    )
    df.loc[df.format == 4, "date"] = pd.to_datetime(
        df.loc[df.format == 4, "time"], format="YR%Y"
    )

    df["frequency"] = np.nan

    df.loc[(df.format == 1) | (df.format == 4), "frequency"] = "yearly"

    df.loc[df.format == 2, "frequency"] = "quarterly"

    df.loc[df.format == 3, "frequency"] = "monthly"

    df.set_index("date", inplace=True)

    del df["time"]

    del df["format"]

    return df
