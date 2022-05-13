import sklearn
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from epigraphhub.models import sklearn_models
import pytest

def lgbm_model(params=None):
    '''
    Return an LGBM model
    :param kwargs:
    :return: LGBMRegressor model
    '''
    if params is None:
        params = {
            'n_jobs': 8,
            'max_depth': 4,
            'max_bin': 63,
            'num_leaves': 255,
#             'min_data_in_leaf': 1,
            'subsample': 0.9,
            'n_estimators': 200,
            'tree_learner': 'feature', 
            'learning_rate': 0.1,
            'colsample_bytree': 0.9,
            'boosting_type': 'gbdt'
        }


    model = lgb.LGBMRegressor(objective='regression', **params)

    return model 

norm_values = [(True),(False)]
@pytest.mark.parametrize("norm", norm_values, ids=["norm=True", "norm=False"])
def test_rolling_predictions_one(get_df_test, norm):

    df = get_df_test
    target_name = "hosp_GE"
    model = lgbm_model()

    df_preds, model, scx, scy = sklearn_models.rolling_predictions_one(model,target_name,
     df, ini_date="2020-03-01", split=0.75, horizon_forecast=2, maxlag=3, norm = norm, norm_model_x =  MinMaxScaler(),
    norm_model_y =  MinMaxScaler(), 
    save = False, path=None, label= 'skl_one'
    )

    df_preds = df_preds.dropna()

    assert not df_preds.empty
    assert df_preds.shape[1] == 3
    assert isinstance(df_preds.index, pd.core.indexes.datetimes.DatetimeIndex)

    assert isinstance(model, lgb.sklearn.LGBMRegressor)

    if norm == True:
        assert isinstance(scx, sklearn.preprocessing._data.MinMaxScaler )
        assert isinstance(scy, sklearn.preprocessing._data.MinMaxScaler )

    else:
        assert scx == None
        assert scy == None

@pytest.mark.parametrize("norm", norm_values, ids=["norm=True", "norm=False"])
def test_rolling_predictions_mult(get_df_test, norm):

    df = get_df_test
    target_name = "hosp_GE"
    model = lgbm_model()
    horizon_forecast = 2

    df_preds, models, scx, scys = sklearn_models.rolling_predictions_mult(model,target_name,
     df, ini_date="2020-03-01", split=0.75, horizon_forecast=horizon_forecast, maxlag=3, norm = norm, norm_model_x =  MinMaxScaler(),
    norm_model_y =  MinMaxScaler(), 
    save = False, path=None, label= 'lgbm_mult'
    )

    df_preds = df_preds.dropna()

    assert not df_preds.empty
    assert df_preds.shape[1] == 3
    assert isinstance(df_preds.index, pd.core.indexes.datetimes.DatetimeIndex)

    assert len(models) == horizon_forecast 

    for i in range(1, len(models) +1 ):
        assert isinstance(models[i], lgb.sklearn.LGBMRegressor)

    if norm == True:
        assert isinstance(scx, sklearn.preprocessing._data.MinMaxScaler )
        for i in range(1, len(scys) +1 ):
            assert isinstance(scys[i], sklearn.preprocessing._data.MinMaxScaler )

    else:
        assert scx == None
        assert scys == dict()


@pytest.mark.parametrize("norm", norm_values, ids=["norm=True", "norm=False"])
def test_training_model_one(get_df_test, norm):
    
    model = lgbm_model()
    df = get_df_test
    target_name = "hosp_GE"
    model, scx, scy = sklearn_models.training_model_one(model,
        target_name,
        df,
        ini_date = '2020-03-01', 
        horizon_forecast=2,
        maxlag=3,
        norm_model_x =  MinMaxScaler(), norm_model_y =  MinMaxScaler(),  norm = norm,
        save = True, path = 'tests/saved_models_for_test/',  label= 'lgbm_one'
    )

    assert isinstance(model, lgb.sklearn.LGBMRegressor)

    if norm == True:
        assert isinstance(scx, sklearn.preprocessing._data.MinMaxScaler )
        assert isinstance(scy, sklearn.preprocessing._data.MinMaxScaler )

    else:
        assert scx == None
        assert scy == None

@pytest.mark.parametrize("norm", norm_values, ids=["norm=True", "norm=False"])
def test_training_model_mult(get_df_test, norm):

    model = lgbm_model()
    df = get_df_test
    target_name = "hosp_GE"

    models, scx, scys = sklearn_models.training_model_mult(
        model,
        target_name,
        df,
        ini_date = '2020-03-01', 
        horizon_forecast=2,
        maxlag=3,
        norm_model_x =  MinMaxScaler(), norm_model_y =  MinMaxScaler(),  norm = norm,
        save = True, path = 'tests/saved_models_for_test',  label= 'lgbm_mult'
    )

    for i in range(1, len(models) +1 ):
        assert isinstance(models[i], lgb.sklearn.LGBMRegressor)

    if norm == True:
        assert isinstance(scx, sklearn.preprocessing._data.MinMaxScaler )
        for i in range(1, len(scys) +1 ):
            assert isinstance(scys[i], sklearn.preprocessing._data.MinMaxScaler )

    else:
        assert scx == None
        assert scys == dict()


@pytest.mark.parametrize("norm", norm_values, ids=["norm=True", "norm=False"])
def test_rolling_forecast_one(get_df_test, norm):

    df = get_df_test

    target_name = "hosp_GE"
    horizon_forecast = 2

    df_for = sklearn_models.rolling_forecast_one(
        target_name, df, horizon_forecast=horizon_forecast, maxlag=3,norm = norm,  path="tests/saved_models_for_test/", label = 'lgbm_one'
    )

    assert not df_for.empty
    assert (df_for.shape[1] == 1) and (df_for.shape[0] == horizon_forecast )
    assert isinstance(df_for.index, pd.core.indexes.datetimes.DatetimeIndex)

@pytest.mark.parametrize("norm", norm_values, ids=["norm=True", "norm=False"])
def test_rolling_forecast_mult(get_df_test, norm):

    df = get_df_test

    target_name = "hosp_GE"
    horizon_forecast = 2

    df_for = sklearn_models.rolling_forecast_mult(
        target_name, df, horizon_forecast=horizon_forecast, maxlag=3,norm = norm,  
        path="tests/saved_models_for_test/", label = 'lgbm_mult'
    )

    assert not df_for.empty
    assert (df_for.shape[1] == 1) and (df_for.shape[0] == horizon_forecast )
    assert isinstance(df_for.index, pd.core.indexes.datetimes.DatetimeIndex)
