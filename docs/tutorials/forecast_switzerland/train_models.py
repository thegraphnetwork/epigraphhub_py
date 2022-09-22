from forecast_swiss import train_all_cantons

target_curve_name = "hosp"
predictors = ["foph_test_d", "foph_cases_d", "foph_hosp_d"]
ini_date = "2020-05-01"

train_all_cantons(
    target_curve_name,
    predictors,
    ini_date=ini_date,
    path="saved_models_dash",
)

target_curve_name = "total_hosp"
predictors = ["foph_test_d", "foph_cases_d", "foph_hosp_d", "foph_hospcapacity_d"]
ini_date = "2020-05-01"

train_all_cantons(
    target_curve_name,
    predictors,
    ini_date=ini_date,
    path="saved_models_dash",
)

target_curve_name = "icu_patients"
predictors = ["foph_test_d", "foph_cases_d", "foph_hosp_d", "foph_hospcapacity_d"]
ini_date = "2020-05-01"

train_all_cantons(
    target_curve_name,
    predictors,
    ini_date=ini_date,
    path="saved_models_dash",
)
