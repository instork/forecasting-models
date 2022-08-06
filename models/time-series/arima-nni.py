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


def get_train_val_sets(full_set, start_date, train_size):
    start_dates = pd.date_range(start_date, start_date+dt.timedelta(days=6), freq='1d')
    train_sets = []
    for start_date in start_dates:
        train_idx = pd.date_range(start_date-dt.timedelta(days=train_size), 
                                  start_date-dt.timedelta(days=1), freq='1d')
        train_sets.append(full_set[full_set.index.isin(train_idx)])
    val_sets = full_set[full_set.index.isin(start_dates)]
    return train_sets, val_sets


def train_evaluate(args):
    file_dir = '../../data/arima'
    os.makedirs(file_dir, exist_ok=True)
    p, d, q, train_size = args['p'], args['d'], args['q'], args['train_size']

    daily_btc_df = pd.read_csv('../../data/daily_btc_df.csv', parse_dates=['etz_date'])
    val_start_trend = pd.read_csv('../../data/val_start_trend.csv', parse_dates=['etz_date'])
    
    daily_btc_df = daily_btc_df.set_index('etz_date')
    daily_btc_series = daily_btc_df.trade_price
    
    full_results = []
    for start_date, val_type in zip(val_start_trend['etz_date'], val_start_trend['trend']):

        convergence_error, stationarity_error = 0, 0
        val_preds_steps = []
        val_preds_seq = None
        aic = []
        bic = []
        train_sets, val_sets = get_train_val_sets(daily_btc_series, start_date, train_size)
        for i, train_set in enumerate(train_sets):
            try:
                model = ARIMA(endog=train_set, order=(p, d, q)).fit()
            except LinAlgError:
                convergence_error += 1
            except ValueError:
                stationarity_error += 1
            except Exception as exception:
                logger.exception(exception)
            forecast = model.forecast(steps=1)
            val_preds_steps.append(forecast)

            aic.append(model.aic)
            bic.append(model.bic)
            if i == 0:
                val_preds_seq = model.forecast(steps=7)
            
        val_preds_steps = pd.concat(val_preds_steps)
        val_preds_steps = val_preds_steps.values
        val_preds_seq = val_preds_seq.values
        val_true = val_sets.values

        rmse_seq = np.sqrt(mean_squared_error(y_true=val_true, y_pred=val_preds_seq))
        rmse_steps = np.sqrt(mean_squared_error(y_true=val_true, y_pred=val_preds_steps))

        cur_results = dict(
                    start_date = start_date,
                    val_type = val_type,
                    rmse_seq = rmse_seq,
                    rmse_steps = rmse_steps,
                    aic = np.mean(aic),
                    bic = np.mean(bic),
                    convergence_error = convergence_error,
                    stationarity_error = stationarity_error,
        )
        full_results.append(cur_results)
        nni.report_intermediate_result(rmse_steps)

    file_name = f'{p}_{d}_{q}_{train_size}.csv'
    file_loc = os.path.join(file_dir, file_name)
    full_results_df = pd.DataFrame(full_results)
    full_results_df.to_csv(file_loc, index=False)
    
    nni.report_final_result(full_results_df['rmse_steps'].mean())

    
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
