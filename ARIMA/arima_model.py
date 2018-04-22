import argparse
import json
import pandas as pd
import numpy as np
import itertools
import operator
import warnings
import pprint
from datetime import datetime
from ..support.io_support import load_data
from ..support.evaluation import smape, evaluate_smape
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
    pages2, date2, visits2 = load_data('data/train_2.csv')

    # fitting and forecasting ARIMA model
    print('***** Fitting and Forecasting ARIMA model *************************')
    itsm = ITSM()
    M = ['log', 'season', 6, 'trend', 1]
    visits_pred = np.array([])
    pages_pred = np.array([])
    expressions = {}; predictions = {}
    for i, series in enumerate(visits):
        print('***** Fitting ARMA for page', i, ': ', pages[i])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
        try:
            model = fit_arima_model(series, M)
            # forecasts
            forecast = itsm.forecast(series, M, model, pred_days)
            # truths
            truth = visits2[pages2==pages[i],550:(550+pred_days)]
            # [truth, pred, lower_b, upper_b]
            kkk = np.vstack((truth, forecast['pred'], forecast['l'], forecast['u'])).T
            print('forecasts: \n', kkk)
            print('smape score: ', smape(truth, forecast['pred']))
            # append predictions and page names
            visits_pred = np.append(visits_pred, forecast['pred'])
            pages_pred = np.append(pages_pred, pages[i])
            # expression of each page (in dictionary):
            # {page id, page name, std, smape score}
            expressions[pages[i]] = {
                'ID': i, 'Page': pages[i],
                'standard deviation': np.std(series), 'SMAPE score': smape(truth, forecast['pred'])}
            json.dump(expressions, open('preds/expressions.json', 'w'), indent=2)
            # forecast of each page (in dictionary):
            # {page name: {pred, lower_b, upper_b}}
            array2list = [(key, value.tolist()) for key, value in list(forecast.items())]
            predictions[pages[i]] = dict(array2list)
            json.dump(predictions, open('preds/predictions.json', 'w'), indent=2)
        except: pass

    # formatting
    print('***** Formatting Predictions **************************************')
    dates_pred = pd.date_range('2017-01-01', periods = pred_days).values
    dates_pred = np.array([str(t)[:10] for t in dates_pred], dtype=object)
    page_date_str = np.array([ p + str('_') + dates_pred for p in pages_pred]).flatten()
    page_visits = pd.DataFrame(data = [page_date_str, visits_pred], index = ['Page', 'Visits']).T
    page_visits.to_csv('preds/prediction_1.csv', index = False)
    submission = pd.merge(page_visits, keys, on = ['Page'])[['Id', 'Visits']]
    print('visits_pred shape: ', visits_pred.shape)
    print('pages_pred shape: ', pages_pred.shape)
    print('page_date_str shape: ', page_date_str.shape)
    print('Submission shape: ', submission.shape)

    # save predictions to csv
    print('***** Saving Predictions ******************************************')
    submission.to_csv('preds/submission_1.csv', index=False)

    print('***** Ending time: ', datetime.now(), '****************************')
