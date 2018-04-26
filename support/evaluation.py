import pandas as pd
import numpy as np
import json

def smape(y_true, y_pred):
    """
    Calculates the smape score for evaluating the forecast values

        Args:
            y_true: True time series
                    type: np.array
            y_pred: Predicted time series
                    type: np.array

        Return:
            score: SMAPE score of the prediction
                   type: float64

    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    score = np.nanmean(diff)
    return score

def evaluate_smape(y_true, y_pred):
    """
    Evaluates forecast values via graphs and generates residuals

        Args:
            y_true: True time series
                    type: np.array
            y_pred: Predicted time series
                    type: np.array

        Return:
            res: residuals array of y_true and y_pred
                 type: np.array

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

        Args:
            y_true: True time series
                    type: np.array
            y_pred: Predicted time series
                    type: np.array

        Return:
            score: RMSE value for forecast values
                   type: float64

    """
    res_sq = (y_pred - y_true) ** 2
    score = np.sqrt(sum(res_sq/len(y_true)))
    return score

def smape_arimaset(n):
    """
    Prints out mean smape scores for each set of predictions
    File preds/performance_i contains performance of the pages
    (includes page id, page name, page sd, page SMAPE)

        Args:
            n: Number of sets
               type: int

        Return:
            df.T: Table contains information for each set and total
                  type: pd.DataFrame
                    - Pages: range of pages index
                    - # of Pages: real number of pages forecasted
                    - Mean SMAPE: mean SMAPE score
            SMAPE_array: SMAPE scores for all pages in n sets
                         type: np.array

    """
    SMAPE_array = np.array([])
    SMAPE_mu = []
    set_number = []; set_name = []
    for i in range(1, n+1):
        performances = json.load(open('preds/performances_'+str(i)+'.json'))
        smapes = np.array([dic['SMAPE score'] for dic in list(performances.values())])
        smapes_mean = smapes.mean()
        # append information for each set
        set_name.append(str(0+400*(i-1))+'-'+str(400+400*(i-1)))
        set_number.append(str(len(smapes)))
        SMAPE_mu.append(smapes_mean)
        SMAPE_array = np.append(SMAPE_array, smapes)
    # append with total information
    set_name.append('Total')
    set_number.append(str(len(SMAPE_array)))
    SMAPE_mu.append(float(SMAPE.array.mean()))
    df = pd.DataFrame(data = [number, SMAPE_mu], index = ['# of pages', 'Mean SMAPE'])
    df.columns = set_name
    df.index.name = 'Pages'
    return df.T, SMAPE_array
