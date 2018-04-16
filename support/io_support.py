import pandas as pd

def load_data(path):
    '''
        Load csv file by its path
        
        Args:
            path: path of csv file
                  type: STRING
            
        Return:
            pages: descriptions of each page
                   type: ndarray, shape: (page_numb,)
        
            dates: descriptions of dates
                   type: ndarray, shape: (date_numb,)
        
            visits: number of visits for certain page on certain day
                    type: ndarray, shape: (page_numb,date_numb)
        '''
    raw_data = pd.read_csv(path)
    
    pages = raw_data[['Page']].as_matrix().flatten()
    dates = raw_data.columns.values[1:]
    visits = raw_data.drop('Page',axis=1).as_matrix()
    
    return pages, dates, visits
