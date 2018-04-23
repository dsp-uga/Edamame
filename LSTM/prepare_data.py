import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import numpy as np

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, window_size=1):
    df = pd.DataFrame(data)
    df_c = df.copy()
    for i in range(window_size):
        df = pd.concat([df, df_c.shift(-(i+1))], axis = 1)
#    df.dropna(axis=0, inplace=True)
    return df.fillna(0)

# create a differenced series
def stationary_data(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train):#, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
#    test = test.reshape(test.shape[0], test.shape[1])
#    test_scaled = scaler.transform(test)
    return scaler, train_scaled#, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]








