import argparse
import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
import logging
from pmdarima.arima import auto_arima
import nni
from nni.utils import merge_parameter

MAX_TRAIN_SIZE = 180
logger = logging.getLogger("imbalace_seg_NNI")

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_size", type=int, default=30, 
        help="# of train days"
    )
    args, _ = parser.parse_known_args()
    return args

def train_evaluate(args):
    file_dir = '../../data/arima_4th'
    os.makedirs(file_dir, exist_ok=True)
    train_size = args['train_size']

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
    orders = []

    for i in range(len(val_full_set)):

        train_set = daily_btc_series.iloc[i:max_train_size+i].iloc[-train_size:]
        model = auto_arima(y=train_set, start_p=1, start_q=1, start_P=1, start_Q=1,
                            max_p=5, max_q=5, max_P=5, max_Q=5,stepwise=True,seasonal=True,
                        trace=False)

        forcast = model.predict(n_periods=1)
        model_summary = model.to_dict()
        cur_order = model_summary['order']
        cur_aic = model_summary['aic']
        cur_bic = model_summary['bic']

        val_preds_steps.append(forcast[0]) 
        orders.append(cur_order)    
        aic.append(cur_aic)
        bic.append(cur_bic)


        if i % 10 == 0 and i > 0:
            y_pred = val_preds_steps
            y_true = val_full_set.iloc[:i+1].values
            rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))
            nni.report_intermediate_result(rmse)
            

    assert len(val_full_set) == len(val_preds_steps) == len(aic) == len(bic)
    
    # val_preds_steps = pd.concat(val_preds_steps)
    result_df = pd.DataFrame(val_full_set)
    result_df.columns = ['y_true']
    result_df['y_pred'] = val_preds_steps
    result_df['aic'] = aic
    result_df['bic'] = bic
    result_df[['p','d','q']] = orders 

    rmse = np.sqrt(mean_squared_error(y_true=result_df.y_true, y_pred=result_df.y_pred))
    nni.report_final_result(rmse)

    file_name = f'autoarima_{train_size}.csv'
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
