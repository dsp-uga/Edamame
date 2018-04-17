import pandas as pd
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dropout, Dense
import time
from math import sqrt
from matplotlib import pyplot
import numpy as np
from sklearn.utils import shuffle

import LSTM.prepare_data as predat


# load dataset
series = pd.read_csv('../data/train_1.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
series = series.iloc[0]
print("original series ",series.shape)
# transform data to be stationary
raw_values = series.values
diff_values = predat.difference(raw_values, 1)

# transform data to be supervised learning
supervised = predat.timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
nrow = round(0.8*supervised_values.shape[0])
print("supervised_values ",supervised_values.shape)

train = supervised_values[:nrow, :]
test = supervised_values[nrow-1:-1,:]
print("train ",train.shape)
print("test ",test.shape)

#train, test = supervised_values[0:-12], supervised_values[-12:]

# transform the scale of the data
scaler, train_scaled, test_scaled = predat.scale(train, test)
print(np.shape(train_scaled),np.shape(test_scaled))

train = pd.DataFrame(train_scaled)
test = pd.DataFrame(test_scaled)

train = shuffle(train)
train_X = train.iloc[:,:-1]
train_y = train.iloc[:,-1]
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]

train_X = train_X.values
train_y = train_y.values
test_X = test_X.values
test_y = test_y.values

train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],1)
test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1)


print("train_X_pre",train_X.shape)
print("train_y_pre",train_y.shape)
print("test_X_pre",test_X.shape)
print("test_y_pre",test_y.shape)

model = Sequential()
model.add(LSTM(input_shape = (1,1), output_dim= 50, return_sequences = True))
model.add(Dropout(0.5))
model.add(LSTM(256))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss="mse", optimizer="sgd")
model.summary()

start = time.time()
model.fit(train_X,train_y,batch_size=10,nb_epoch=10,validation_split=0.1)
print("> Compilation Time : ", time.time() - start)

# Doing a prediction on all the test data at once
prediction = model.predict(test_X)
print("preds ",prediction.shape)
print("test_y ",test_y.shape)
test_y = test_y.reshape(test_y.shape[0],1)
print(prediction)
Xt, yt = test_scaled[:, 0], test_scaled[:, 1]

collection = []
for i in range(len(Xt)):
    X = [Xt[i]]
    yhat = prediction[i]
    yhat = predat.invert_scale(scaler, X, yhat)
    yhat = predat.inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    collection.append(yhat)
    expected = raw_values[len(train) + i + 1]
    print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
print(np.shape(collection))
print(np.shape(raw_values[nrow:]))
rmse = sqrt(mean_squared_error(raw_values[nrow:-1], collection))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(raw_values[nrow:-1])
pyplot.plot(collection)
pyplot.show()









