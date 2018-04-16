import numpy as np
import support.io_support as io

path = '../data/train_1.csv'

def test_load_data():
    
    page, date, data = io.load_data(path)
    
    assert np.shape(page) == (145063,)
    assert np.shape(date) == (550,)
    assert np.shape(data) == (145063, 550)
