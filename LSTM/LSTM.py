import pandas as pd
import numpy as np
#from math import sqrt
#import time
from sklearn.preprocessing import MinMaxScaler

from keras.layers import LSTM, Activation, Dropout, Dense
from keras.models import Sequential

#from sklearn.utils import shuffle
#from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot
import LSTM.prepare_data as pda
import support.evaluation as se

series_all = pd.read_csv('../data/train_1.csv')
series_all = series_all.fillna(0)

series_all = series_all.iloc[:,1:]

series2_all = pd.read_csv('../data/train_2.csv')
series2_all = series2_all.fillna(0)
series2_all = series2_all.iloc[:,1:]

train_raw_values = series_all
test_raw_values = series2_all
#y_raw_norm_inverse = test_raw_values.iloc[:, -60:]

train_norm_values = np.log1p(train_raw_values*0.5).astype('float32')
test_norm_values = np.log1p(test_raw_values*0.5).astype('float32')
#train_norm_values = train_raw_values
#test_norm_values = test_raw_values
#print("train_norm_values: ",train_norm_values.shape)
#scalerX, train_norm_values = pda.scale(train_norm_values.values)
#scalery, tmp = pda.scale(y_raw_norm_inverse.values)

#train_norm_values = pd.DataFrame(train_norm_values)

#train_X_norm = train_norm_values.iloc[:, :490]
train_X_norm = train_norm_values#.iloc[:, :488]
train_y_norm = train_norm_values.iloc[:, -62:]

#test_X_norm = train_norm_values.iloc[:, 60:550]
test_X_norm = train_norm_values.iloc[:, -488:]
#test_y_norm = test_norm_values.iloc[:, 550:610]
test_y_norm = test_raw_values.iloc[:, 550:612]

input_dim = test_X_norm.shape[1]
output_dim = test_y_norm.shape[1]

model_lstm2 = Sequential()
model_lstm2.add(LSTM(input_shape=(1,input_dim,), units=50, return_sequences = True))
model_lstm2.add(Dropout(0.5))
model_lstm2.add(LSTM(100))
model_lstm2.add(Dropout(0.5))
#model_lstm2.add(Dense(50))
#model_lstm2.add(Dropout(0.5))
model_lstm2.add(Dense(output_dim))
model_lstm2.add(Activation("linear"))
model_lstm2.compile(loss="mse", optimizer="adam")
model_lstm2.summary()

lstm2_input = train_X_norm.values
lstm2_input = lstm2_input.reshape(145063,1,488)

batch_size = 5000
model_lstm2.fit(lstm2_input, train_y_norm.values, batch_size=batch_size, epochs=20,validation_split=0.1)

lstm2_test_input = test_X_norm.values
lstm2_test_input = lstm2_test_input.reshape(145063,1,488)
y_lstm2_pred_norm = model_lstm2.predict(lstm2_test_input, batch_size=batch_size)



y_lstm2_pred_norm_inverse = np.expm1(y_lstm2_pred_norm)/0.5
np.save('predict_62.npy',y_lstm2_pred_norm_inverse)
#y_lstm2_pred_norm = pd.DataFrame(y_lstm2_pred_norm)
#whole_norm = pd.concat([test_X_norm,y_lstm2_pred_norm],axis=1)
#whole = scalery.inverse_transform(y_lstm2_pred_norm.values)
#y_lstm2_pred_norm_inverse = whole[:,-60:]
#y_lstm2_pred_norm_inverse = pd.DataFrame(whole)

#print(y_lstm2_pred_norm_inverse.shape,y_raw_norm_inverse.shape)
#print(y_lstm2_pred_norm_inverse)
#print(se.smape(y_raw_norm_inverse, y_lstm2_pred_norm_inverse))
print(se.smape(test_y_norm, y_lstm2_pred_norm_inverse))

#tmp = pd.DataFrame(tmp)
#tmp_inverse = scalery.inverse_transform(tmp)
#tmp_inverse = pd.DataFrame(tmp_inverse)
#print(tmp.head())
#print("****************************")
#print(y_lstm2_pred_norm.head())


pyplot.figure(0)
#pyplot.plot(range(60),y_lstm2_pred_norm_inverse.iloc[38], label = "predict")
#pyplot.plot(range(60),y_raw_norm_inverse.iloc[38], label = "raw")
pyplot.plot(range(62),y_lstm2_pred_norm_inverse[38], label = "predict")
pyplot.plot(range(62),test_y_norm.iloc[38], label = "raw")
pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
pyplot.show()




