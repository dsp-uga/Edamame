import numpy as np
import json
import matplotlib.pyplot as plt
from ARIMA.itsmpy import ITSM
from io_support import load_data
from evaluation import smape

def fig_detrend(page, true_data, path=None, p=2):
    """
        Visualizes two figures in one window with trend series and detrend series

        Args:
            page: page name to visualize
                  type: str
            true_data: time series data load by load_data(), [pages, dates, visits]
                       type: list
            path: directory to save the figure, default = None
                  type: str
            p: polynomial order (1=linear, 2=quadratic)
               type: int

    """
    pages, dates, visits = true_data
    series = visits[pages==page,:].flatten()
    time = len(dates)
    itsm = ITSM()
    trendc = itsm.trend(series, p)
    plt.style.use('ggplot')
    plt.figure(figsize = (20,8))
    # trend series
    plt.subplot(1,2,1)
    plt.plot(series, color="#2980B9")
    plt.plot(trendc, color="#EC7063",linewidth=3)
    plt.xticks(np.arange(0,time,time//6), dates[np.arange(0,time,time//6)])
    plt.tick_params(labelcolor = '#239B56', labelsize = 12, width = 3, pad = 12)
    plt.xlabel("Dates", fontsize = 20, labelpad = 14)
    plt.ylabel("Visits", fontsize = 20, labelpad = 14)
    plt.title('Visits with trend components', fontsize = 26, pad = 16)
    plt.legend(['original data', 'trend components'], fontsize=18)
    # detrend series
    plt.subplot(1,2,2)
    plt.plot(series, color="#2980B9")
    plt.plot(series-trendc, color="#16A085")
    plt.xticks(np.arange(0,time,time//6), dates[np.arange(0,time,time//6)])
    plt.tick_params(labelcolor = '#239B56', labelsize = 12, width = 3, pad = 12)
    plt.xlabel("Dates", fontsize = 20, labelpad = 14)
    plt.ylabel("Visits", fontsize = 20, labelpad = 14)
    plt.title('Visits with/without trend components', fontsize = 26, pad = 16)
    plt.legend(['trend', 'detrend'], fontsize=18)
    if path != None:
        plt.savefig(path+'detrend.png')
    plt.show()

def fig_deseasonal(page, true_data, path=None, d=2):
    """
        Visualizes two figures in one window with seasonal series and deseasonal series

        Args:
            page: page name to visualize
                  type: str
            true_data: time series data load by load_data(), [pages, dates, visits]
                       type: list
            path: directory to save the figure, default = None
                  type: str
            d: number of observation per season
               type: int

    """
    pages, dates, visits = true_data
    series = visits[pages==page,:].flatten()
    time = len(dates)
    itsm = ITSM()
    seasonc = itsm.season(series, d)
    plt.style.use('ggplot')
    plt.figure(figsize = (20,8))
    # trend series
    plt.subplot(1,2,1)
    plt.plot(series, color="#2980B9")
    plt.plot(seasonc, color="#EC7063",linewidth=3)
    plt.xticks(np.arange(0,time,time//6), dates[np.arange(0,time,time//6)])
    plt.tick_params(labelcolor = '#239B56', labelsize = 12, width = 3, pad = 12)
    plt.xlabel("Dates", fontsize = 20, labelpad = 14)
    plt.ylabel("Visits", fontsize = 20, labelpad = 14)
    plt.title('Visits with seasonal components', fontsize = 26, pad = 16)
    plt.legend(['original data', 'seasonal components'], fontsize=18)
    # detrend series
    plt.subplot(1,2,2)
    plt.plot(series, color="#2980B9")
    plt.plot(series-seasonc, color="#16A085")
    plt.xticks(np.arange(0,time,time//6), dates[np.arange(0,time,time//6)])
    plt.tick_params(labelcolor = '#239B56', labelsize = 12, width = 3, pad = 12)
    plt.xlabel("Dates", fontsize = 20, labelpad = 14)
    plt.ylabel("Visits", fontsize = 20, labelpad = 14)
    plt.title('Visits with/without seasonal components', fontsize = 26, pad = 16)
    plt.legend(['seasonal', 'deseasonal'], fontsize=18)
    if path != None:
        plt.savefig(path+'deseaonl.png')
    plt.show()

def fig_season_trend(page, true_data, path=None, d=7, p=2):
    """
        Visualizes two figures in one window with seasonal and trend components

        Args:
            page: page name to visualize
                  type: str
            true_data: time series data load by load_data(), [pages, dates, visits]
                       type: list
            path: directory to save the figure, default = None
                  type: str
            d: number of observation per season
               type: int
            p: polynomial order (1=linear, 2=quadratic)
               type: int

    """
    pages, dates, visits = true_data
    series = visits[pages==page,:].flatten()
    time = len(dates)
    itsm = ITSM()
    seasonc = itsm.season(series, d)
    trendc = itsm.trend(series, p)
    ### figure
    plt.style.use('ggplot')
    plt.figure(figsize = (24,9))
    #### seasonal components
    plt.subplot(1,2,1)
    plt.plot(np.arange(time), series, color = "#2980B9")
    plt.plot(np.arange(time), seasonc, color = "#48C9B0", linewidth = 2)
    plt.xticks(np.arange(0,time,time//6), dates[np.arange(0,time,time//6)])
    plt.tick_params(labelcolor = '#239B56', labelsize = 12, width = 3, pad=12)
    plt.xlabel("Dates", fontsize = 20, labelpad = 14)
    plt.ylabel("Visits", fontsize = 20, labelpad = 14)
    plt.legend(['original data', 'seasonal components'], fontsize=18)
    plt.title('Visits with seasonal components', fontsize = 26, pad = 16)
    #### trend components
    plt.subplot(1,2,2)
    plt.plot(np.arange(time), series, color = "#2980B9")
    plt.plot(np.arange(time), trendc, color = "#48C9B0", linewidth = 3)
    plt.xticks(np.arange(0,time,time//6), dates[np.arange(0,time,time//6)])
    plt.tick_params(labelcolor = '#239B56', labelsize = 12, width = 3, pad=12)
    plt.xlabel("Dates", fontsize = 20, labelpad = 14)
    plt.ylabel("Visits", fontsize = 20, labelpad = 14)
    plt.legend(['original data', 'trend components'], fontsize=18)
    plt.title('Visits with trend components', fontsize = 26, pad = 16)
    #### big picture
    plt.suptitle('Page: ' + page, fontsize=30)
    plt.subplots_adjust(top=0.85)
    if path != None:
        plt.savefig(path+'season_trend_fig.png')
    plt.show()

def fig_forecast(page, true_data, train_len, pred, pred2=None, bound=None, path=None):
    """
        Visualizes two figures in one window with long-term and short-term forecasts
        (Only for)

        Args:
            page: page name to visualize
                  type: str
            true_data: time series data load by load_data(), [pages, dates, visits]
                       contains the true values of forecasting dates
                       type: list
            train_len: length of training set of forecasts
                       type: int
            pred: forecasts of the page
                  type: np.array
            pred2: forecasts 2 of the page, default = None
                   type: np.array
            bound: dictionary of upper and lower bounds, default = None
                   tpye: dictionary
            path: directory to save the figure, default = None
                  type: str

    """
    days = len(pred)
    pages, dates, visits = true_data
    series = visits[pages==page, :(train_len+days)].flatten()
    score = smape(series[train_len, (train_len+days)], pred).round(decimals=4)
    if pred != None:
        score2 = smape(series[train_len, (train_len+days)], pred2).round(decimals=4)

    ### figure
    plt.style.use('ggplot')
    plt.figure(figsize = (24,9))
    # long-term
    plt.subplot(1,2,1)
    plt.plot(np.arange((train_len+days)), series[:(train_len+days)], '-o', color = "#2980B9", markersize = 4)
    plt.plot(np.arange(train_len,(train_len+days)), pred, '-o', color = "#EC7063", markersize = 4)
    if pred2 != None:
        plt.plot(np.arange(train_len,(train_len+days)), pred2, '-o', color = "#16A085", markersize = 4)
    plt.xticks(np.arange(0,train_len+days,(train_len+days)//6), dates[np.arange(0,train_len+days,(train_len+days)//6)])
    plt.tick_params(labelcolor='#239B56', labelsize = 12, width = 3, pad=12)
    plt.xlabel("Dates", fontsize = 20, labelpad = 14)
    plt.ylabel("Visits", fontsize = 20, labelpad = 14)
    if pred2 == None:
        plt.legend(['original data', 'forecast'], fontsize=18)
    else:
        plt.legend(['original data', 'forecast 1', 'forecast 2'], fontsize=18)
    plt.title('Visits from ' + dates[0] + ' to ' + dates[(train_len+days-1)], fontsize=26)
    # short-term
    plt.subplot(1,2,2)
    plt.plot(np.aranget(train_len,(train_len+days)), series[train_len:(train_len+days)], '-o', color = "#2980B9", markersize = 4)
    plt.plot(np.arange(train_len,(train_len+days)), pred, '-o', color = "#EC7063", markersize = 4)
    if pred2 != None:
        plt.plot(np.arange(train_len,(train_len+days)), pred2, '-o', color = "#16A085", markersize = 4)
    if bound != None:
        lower = bound['l']
        upper = bound['u']
        plt.plot(np.arange(train_len,(train_len+days)), lower, linestyle = 'dashed', color = "#EC7063")
        plt.plot(np.arange(train_len,(train_len+days)), upper, linestyle = 'dashed', color = "#EC7063")
    plt.title('Visits from ' + dates[train_len] + ' to ' + dates[(train_len+days-1)], fontsize=26)
    plt.xticks(np.arange(train_len,train_len+days,(train_len+days)//6), dates[np.arange(train_len,train_len+days,(train_len+days)//6)])
    plt.xlabel("Dates", fontsize = 20, labelpad = 14)
    plt.ylabel("Visits", fontsize = 20, labelpad = 14)
    plt.tick_params(labelcolor='#239B56', labelsize = 12, width = 3, pad=12)
    if pred2 == None:
        plt.legend(['original data', 'forecast'], fontsize=18)
        plt.suptitle('Page: ' + page + '\n(SMAPE score: ' + str(score), fontsize=30)
    else:
        plt.legend(['original data', 'forecast 1', 'forecast 2'], fontsize=18)
        plt.suptitle('Page: ' + page +
                     '\n(SMAPE score: ' + str(score) + ' for forecast 1, ' + str(score2) + ' for forecast 2)',
                     fontsize=30)
    plt.subplots_adjust(top=0.8)
    if path != None:
        plt.savefig(path+'forecast_fig.png')
    plt.show()

def fig_smape_distribution(s, path=None):
    """
        Visualizes the distribution of SMAPE scores of pages

        Args:
            s: SMAPE scores of pages
               type: np.array
            path: directory to save the figure, default = None
                  type: str

    """
    plt.style.use('ggplot')
    plt.figure(figsize=(18,12))
    plt.hist(s, bins=50, color="#16A085")
    plt.tick_params(labelcolor='#239B56', labelsize=16, width=3, pad=12)
    plt.xlabel("SMAPE score", fontsize=20, labelpad=14)
    plt.ylabel("Frequency", fontsize=20, labelpad=14)
    plt.title('Distribution of SMAPE score', fontsize=26, pad=16)
    if path != None:
        plt.savefig(path+'smape_dist_fig.png')
    plt.show()
