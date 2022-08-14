import argparse
import os
import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from statsmodels.tsa.arima.model import ARIMA
import datetime as dt
from sklearn.metrics import mean_squared_error
import logging

import nni
from nni.utils import merge_parameter

MAX_TRAIN_SIZE = 180
logger = logging.getLogger("imbalace_seg_NNI")

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p", type=int, default=1, 
        help="p on ARIMA(p,d,q)"
    )
    parser.add_argument(
        "--d", type=int, default=1, 
        help="d of ARIMA(p,d,q)"
    )
    parser.add_argument(
        "--q", type=int, default=1, 
        help="q of ARIMA(p,d,q)"
    )
    parser.add_argument(
        "--train_size", type=int, default=30, 
        help="# of train days"
    )
    args, _ = parser.parse_known_args()
    return args

def train_evaluate(args):
    file_dir = '../../data/arima_3rd'
    os.makedirs(file_dir, exist_ok=True)
    p, d, q, train_size = args['p'], args['d'], args['q'], args['train_size']

    daily_btc_df = pd.read_csv('../../data/daily_btc_df.csv', parse_dates=['etz_date'])
    daily_btc_df = daily_btc_df.set_index('etz_date')
    daily_btc_series = daily_btc_df.trade_price

    if not daily_btc_series.index.is_monotonic_increasing:
        daily_btc_series = daily_btc_series.sort_index()
    
    max_train_size = MAX_TRAIN_SIZE
    val_full_set = daily_btc_series.iloc[max_train_size:]

    aic = []
    bic = []
    val_preds_steps = []
    convergence_error, stationarity_error = 0, 0
    
    for i in range(len(val_full_set)):
        train_set = daily_btc_series.iloc[i:max_train_size+i].iloc[-train_size:]
        try:
            model = ARIMA(endog=train_set, order=(p, d, q)).fit()
            forecast = model.forecast(steps=1)
            cur_aic = model.aic
            cur_bic = model.bic

        except LinAlgError:
            new_idx = daily_btc_series.iloc[max_train_size+i:max_train_size+i+1].index[0] + dt.timedelta(days=1)
            forecast = pd.Series(np.nan, index=[new_idx])
            cur_aic = np.nan
            cur_bic = np.nan
            convergence_error += 1
            logger.info(f"LinAlgError {convergence_error} {new_idx}")

        except ValueError:
            new_idx = daily_btc_series.iloc[max_train_size+i:max_train_size+i+1].index[0] + dt.timedelta(days=1)
            forecast = pd.Series(np.nan, index=[new_idx])
            cur_aic = np.nan
            cur_bic = np.nan
            stationarity_error += 1
            logger.info(f"stationarity_error {stationarity_error} {new_idx}")

        val_preds_steps.append(forecast)

        aic.append(cur_aic)
        bic.append(cur_bic)

        if i % 10 == 0 and i > 0:
            y_pred = pd.concat(val_preds_steps).values
            y_true = val_full_set.iloc[:i+1].values
            nni.report_intermediate_result(np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred)))

    assert len(val_full_set) == len(val_preds_steps) == len(aic) == len(bic)
    
    val_preds_steps = pd.concat(val_preds_steps)
    result_df = pd.concat([val_full_set, val_preds_steps], axis=1)
    result_df.columns = ["y_true", "y_pred"]
    result_df["aic"] = aic
    result_df["bic"] = bic

    rmse = np.sqrt(mean_squared_error(y_true=result_df.y_true, y_pred=result_df.y_pred))

    nni.report_final_result(rmse)

    file_name = f'{p}_{d}_{q}_{train_size}.csv'
    file_loc = os.path.join(file_dir, file_name)
    result_df.to_csv(file_loc, index=False)

    
if __name__ == '__main__':
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        train_evaluate(params)
    except Exception as exception:
        logger.exception(exception)
        raise
