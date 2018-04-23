import pandas as pd
import numpy as np

from keras import regularizers
from keras.layers.normalization import BatchNormalization
import keras

from keras.layers import Input, LSTM, Activation, Dropout, Dense
from keras.models import Sequential, Model

from matplotlib import pyplot
import support.evaluation as se

series_all = pd.read_csv('../data/train_1.csv')
series_all = series_all.fillna(0)

series_all = series_all.iloc[:,1:]

series2_all = pd.read_csv('../data/train_2.csv')
series2_all = series2_all.fillna(0)
series2_all = series2_all.iloc[:,1:]

train_raw_values = series_all
test_raw_values = series2_all
y_raw_norm_inverse = test_raw_values.iloc[:, -60:]

train_norm_values = np.log1p(train_raw_values).astype('float32')
test_norm_values = np.log1p(test_raw_values).astype('float32')

train_X_norm = train_norm_values.iloc[:, :490]
train_y_norm = train_norm_values.iloc[:, -60:]

test_X_norm = train_norm_values.iloc[:, 60:550]
test_y_norm = test_norm_values.iloc[:, 550:610]

print("shape: ")
print(train_X_norm.shape,train_y_norm.shape,test_X_norm.shape,test_y_norm.shape)

input_dim = test_X_norm.shape[1]
output_dim = test_y_norm.shape[1]

dropout = 0.5
regularizer = 0.00004

model = Sequential()
model.add(Dense(100, input_shape=(input_dim,), activation='relu',kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer)))
model.add(Dropout(dropout))
model.add(Dense(50, activation='relu',kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer)))
model.add(BatchNormalization(beta_regularizer=regularizers.l2(regularizer),gamma_regularizer=regularizers.l2(regularizer)))
model.add(Dropout(dropout))
model.add(Dense(20, activation='relu',kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer)))
model.add(Dropout(dropout))
model.add(Dense(25, activation='relu',kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer)))
model.add(Dropout(dropout))
model.add(Dense(output_dim, activation='linear',kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer)))
model.compile(loss='mse', optimizer='adam')
model.summary()

batch_size = 5000
model.fit(train_X_norm.values, train_y_norm.values, batch_size=batch_size, epochs=20,validation_split=0.1)

y_pred_norm = model.predict(test_X_norm.values, batch_size=batch_size)

y_pred_norm_inverse = np.expm1(y_pred_norm)
print(se.smape(y_raw_norm_inverse, y_pred_norm_inverse))


