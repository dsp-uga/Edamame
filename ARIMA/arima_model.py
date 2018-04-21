import argparse
import pandas as pd
import numpy as np
import itertools
import operator
import warnings
import pprint
from datetime import datetime
from ..support.io_support import load_data
# from .evaluation import smape, evaluate_smape
from .itsmpy import ITSM

def fit_arima_model(series, M):
    """
    Fits best arima model by lowest aicc values
    :type series: np.arrays, time series data
    :type M: list, data model ([] for None)
    :rtype a: dictionary, ARMA model coefficients (phi, theta, sigma2)
    """
    itsm = ITSM()
    e = itsm.Resid(series, M)
    warnings.filterwarnings("ignore")
    a = itsm.autofit(e)
    print('ARMA model: ')
    pprint.pprint(a, width=1)
    return a

def main(data_path, keys_path, pred_days=60):
    print('***** Starting time: ', datetime.now(), '**************************')

    print('***** Loading data ************************************************')
    pages, dates, visits = load_data(data_path)
    # pages, dates, visits = pages[:1], dates, visits[:1,:]
    keys = pd.read_csv(keys_path)
    print('Visits data shape: ', visits.shape)
    print('pages shape: ', pages.shape)
    print('dates shape: ', dates.shape)
    print('Key data shape: ', keys.shape)

    # fill missing values
    # print('***** Filling Missing Values **************************************')
    # train_filled = fillna_forward_backward(train)

    # fitting and forecasting ARIMA model
    print('***** Fitting and Forecasting ARIMA model *************************')
    itsm = ITSM()
    M = ['log', 'season', 6, 'trend', 1]
    visits_pred = np.array([])
    for i, series in enumerate(visits):
        print('***** Fitting ARMA for page', i, ': ', pages[i])
        model = fit_arima_model(series, M)
        forecast = itsm.forecast(series, M, model, pred_days)
        kkk = np.vstack((forecast['pred'], forecast['l'], forecast['u'])).T
        print('forecasts: \n', kkk)
        visits_pred = np.append(visits_pred, forecast['pred'])

    # formatting
    print('***** Formatting Predictions **************************************')
    dates_pred = pd.date_range('2017-01-01', periods = pred_days).values
    dates_pred = np.array([str(t)[:10] for t in dates_pred], dtype=object)
    page_date_str = np.array([ p + str('_') + dates_pred for p in pages]).flatten()
    page_visits = pd.DataFrame(data = [page_date_str, visits_pred], index = ['Page', 'Visits']).T
    page_visits.to_csv('preds/prediction_1.csv', index = False)
    submission = pd.merge(page_visits, keys, on = ['Page'])[['Id', 'Visits']]
    print('Submission shape: ', submission.shape)
    print('Submission preview: \n', submission.head())

    # save predictions to csv
    print('***** Saving Predictions ******************************************')
    submission.to_csv('preds/submission_1.csv', index=False)

    print('***** Ending time: ', datetime.now(), '****************************')
