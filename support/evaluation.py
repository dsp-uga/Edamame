import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def smape(y_true, y_pred):
    """
        Calculates the smape score for evaluating the forecast values
        :type y_true: np.array
        :type y_pred: np.array
        :rtype score: np.float64
        """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    score = np.nanmean(diff)
    return score

def evaluate_smape(y_true, y_pred):
    """
        Evaluates forecast values via graphs
        Generates residuals
        :type y_true: np.array
        :type y_pred: np.array
        :rtype res: np.array
        """
    x = np.linspace(0,10,1000)
    res = [smape(y_true, i * y_pred) for i in x]
    plt.plot(x, res)
    plt.show()
    print('SMAPE min:%0.2f' % np.min(res), ' at %0.2f' % x[np.argmin(res)])
    print('SMAPE is :%0.2f' % smape(y_true, y_pred*np.nanmedian(y_true)),
          ' at median %0.2f' % np.nanmedian(y_true))
    return res

def rmse(y_true, y_pred):
    """
        Calculates the RMSE value for forecast Values
        :type y_true: np.array
        :type y_pred: np.array
        :rtype score: np.float64
        """
    res_sq = (y_pred - y_true) ** 2
    score = np.sqrt(sum(res_sq/len(y_true)))
    return score
