import pandas as pd
import wbgapi as wb


def get_pop_data(country, last_years=20, fx_et="5Y"):

    """
    Function to get the population data stratified by age and sex from
    the world data bank

    :params country: string. ISO-CODE of the country of interest.
    :params last_years: int. Number of last years whose you want to get
                         the population data.
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

    df = wb.data.DataFrame(series=ind, economy=country, mrv=last_years)

    df = df.T

    new_cols = []

    for i in df.columns:

        new_cols.append(((i[3:-3]).lower()).replace(".", "_"))

    df.columns = new_cols

    df.index = df.index.str.strip(to_strip="YR")

    df.index = pd.to_datetime(df.index, format="%Y")
    return df
