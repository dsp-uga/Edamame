import support.data_preprocessing as dpre
import support.io_support as io

from keras.layers import LSTM, Activation, Dropout, Dense
from keras.models import Sequential

def build_LSTM(units, input_dim, output_dim, dropout=0.5, lstm_layers=0, lstm_neurons=[], dense_layers=0, dense_neurons=[]):
    '''
        Build the LSTM model.
        
        Args:
            units:          Number of units in LSTM model
                            type: INT
            input_dim:      Input feature dimension
                            type: INT
            output_dim:     Output dimension
                            type: INT
            dropout:        Dropout number
                            type: INT, default=0.5
            lstm_layers:    Number of additional LSTM layers
                            type: INT, default=0
            lstm_neurons:   List of neuron numbers in each additional LSTM layer
                            type: List, default=[]
            dense_layers:   Number of additional Dense layers
                            type: INT, default=0
            dense_neurons:  List of neuron numbers in each additional Dense layer
                            type: List, default=[]
        
        Return:
            model:          LSTM model
                            type: Keras Sequential Model
        '''
    model = Sequential()
    model.add(LSTM(input_shape=(1,input_dim,), units=units, return_sequences = True))
    model.add(Dropout(dropout))
    if lstm_layers>1 and len(neurons)==lstm_layers:
        for i in range(lstm_layers):
            model.add(LSTM(neurons[i]))
            model.add(Dropout(dropout))
    if dense_layers>0 and len(dense_neurons)==dense_layers:
        model.add(Dense(dense_neurons[i]))
        model.add(Dropout(dropout))
    model.add(Dense(output_dim))
    model.add(Activation("linear"))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    
    return model

def fit_predict(model, train_X, train_y, test_X, batch_size=5000, epochs=20, validation_split=0.1):
    '''
        This is for fitting the model and making forecast.
        
        Args:
            model:              the build LSTM model
                                type: Keras Sequential Model
            train_X:            the training X
                                type: ndarray, shape: (series, train_step)
            train_y:            the training y
                                type: ndarray, shape: (series, pred_step)
            test_X:             the testing X
                                type: ndarray, shape: (series, train_step)
            batch_size:         batch size for fitting the model and predicting
                                type: INT, default: 5000
            epochs:             number of epochs to train the model
                                type: INT, default: 20
            validation_split:   ration of training data being splited into validation set
                                type: ndarray, shape: (series, train_step), default: 0.1
                            
        Return:
            pred_norm:          the prediction being made
                                type: ndarray, shape: (serise, pred_step)
        '''
    
    model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    pred_norm = model.predict(test_X, batch_size=batch_size)

    return pred_norm

def main(train_path, test_path, pred_days=60):
    '''
        Main for __main__.py to call
        
        Args:
            train_path:     path to training data
                            type: STRING
            test_path:      path to testing data
                            type: STRING
            pred_days:      number of days to predict
                            type: INT, defalut: 60
        '''
    pages, dates, series1_all = io.load_data(train_path)
    train_raw_values = dpre.clean_Data(series1_all)
    train_norm_values = dpre.normalise_transform(train_raw_values)

    pages, dates, series2_all = io.load_data(test_path)
    test_raw_values = dpre.clean_Data(series2_all)

    train_X, train_y, test_X, test_y = dpre.split_data(train_norm_values,test_raw_values)
    input_dim = test_X.shape[2]
    output_dim = test_y.shape[2]

    model = build_LSTM(5,input_dim, output_dim)
    pred_norm = fit_predict(model, train_X, train_y, test_X, batch_size=5000,epochs=2)

    pred = dpre.normalise_reverse(pred_norm)
    pred.save('lstm_prediction.npy')
