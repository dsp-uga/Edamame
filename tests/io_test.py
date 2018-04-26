import numpy as np
import support.io_support as io

path = '../data/train_1.csv'

def test_load_data():
    
    page, date, data = io.load_data(path)
    
    # All of these loads should succeed without error.
    assert np.shape(page) == (145063,)       # 145063 wiki pages
    assert np.shape(date) == (550,)          # 550 days in train_1
    assert np.shape(data) == (145063, 550)
