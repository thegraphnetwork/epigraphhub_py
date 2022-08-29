import config
from forecast_swiss import forecast_all_cantons, save_to_database
from sqlalchemy import create_engine

engine = create_engine(config.DB_URI)

df_for_hosp = forecast_all_cantons(
    "hosp",
    ["foph_test_d", "foph_cases_d", "foph_hosp_d"],
    vaccine=True,
    smooth=True,
    path="saved_models_dash",
)

save_to_database(df_for_hosp, "ngboost_forecast_hosp_d_results", engine=engine)

df_for_total_hosp = forecast_all_cantons(
    "total_hosp",
    ["foph_test_d", "foph_cases_d", "foph_hosp_d", "foph_hospcapacity_d"],
    vaccine=True,
    smooth=True,
    path="saved_models_dash",
)

save_to_database(
    df_for_total_hosp, "ngboost_forecast_total_hosp_d_results", engine=engine
)

df_for_icu = forecast_all_cantons(
    "icu_patients",
    ["foph_test_d", "foph_cases_d", "foph_hosp_d", "foph_hospcapacity_d"],
    vaccine=True,
    smooth=True,
    path="saved_models_dash",
)

save_to_database(df_for_icu, "ngboost_forecast_total_hosp_d_results", engine=engine)
