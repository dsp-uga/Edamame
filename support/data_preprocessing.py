import numpy as np
import pandas as pd

def clean_Data(data, fill=0, with_pages = False):
    '''
        Clean the input data, including fill nan with 0 and removing page column
        
        Args:
            data:       input data for clean
                        type: ndarray, shape: (series, time_step)
            fill:       the number to fill
                        type: INT
            with_pages: check whether it has page column
                        type: Boolean
        
        Return:
            data:       clean data
                        type: ndarray, shape: (series, time_step)
        
        '''
    data = pd.DataFrame(data)
    data = data.fillna(0).values
    if with_pages:
        data = data[:,1:]
    return data

def normalise_transform(data):
    '''
        This is for normalising the input data.
        
        Args:
            data:               input data to transform
                                type: ndarray, shape: (series, time_step)

        Return:
            transformed_Data:   transformed data
                                type: ndarray, shape: (series, time_step)
        '''
    transformed_Data = np.log1p(data*0.5).astype('float32')
    return transformed_Data

def normalise_reverse(data):
    '''
        This is for rescaling the scaled data.
        
        Args:
            data:           input data to rescale
                            type: ndarray, shape: (series, time_step)
        
        Return:
            reversed_data:  rescaled data
                            type: ndarray, shape: (series, time_step)
        '''
    reversed_data = np.expm1(data)/0.5
    return reversed_data

def split_data(train, test, pred_days=60):
    '''
        This is for spliting raw data into train_X, train_y, test_X, and test_y
        
        Args:
            train:      raw train data
                        type: ndarray, shape: (series, time_step)
            test:       raw test data
                        type: ndarray, shape: (series, time_step)
            pred_day:   number of days to forecast
                        type: INT
            
        Return:
            train_X:    data of training input
                        type: ndarray, shape: (series, time_step)
            train_y:    data of training output
                        type: ndarray, shape: (series, pred_days)
            test_X:     data of testing input
                        type: ndarray, shape: (series, time_step)
            test_y:     raw data for validating testing (ground truth)
                        type: ndarray, shape: (series, pred_days)
        '''
    series_numb, total_length = train.shape

    train_X = train[:, :(total_length - pred_days)]
    print(train_X.shape)
    train_y = train[:, -(pred_days):]
    print(train_y.shape)
    
    test_X = train[:, pred_days:total_length]
    print(test_X.shape)
    test_y = test[:, total_length:(total_length+pred_days)]
    print(test_y.shape)
    
    series, train_step = train_X.shape
    series, pred_step = train_y.shape
    print(train_step,pred_step)
    
    
    train_X = train_X.reshape(series,1,train_step)
    train_y = train_y.reshape(series,1,pred_step)
    test_X = test_X.reshape(series,1,train_step)
    test_y = test_y.reshape(series,1,pred_step)

    return train_X, train_y, test_X, test_y
