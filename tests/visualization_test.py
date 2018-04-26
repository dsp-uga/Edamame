import numpy as np
import matplotlib.pyplot as plt
import ..support.visualization as vis
from ..support.evaluation import smape_arimaset

pages, dates, visits = load_data('data/train_2.csv')
target = 'Special:Search_zh.wikipedia.org_all-access_spider'
pred = np.array([
        622.17393343,  616.0573541 ,  673.81149476,  691.59463221,
        687.23464779,  747.54640094,  736.80742826,  704.74728259,
        737.79333035,  772.54359789,  764.94989943,  802.51747608,
        804.43017028,  779.11397926,  790.41399787,  833.63083721,
        843.07646603,  862.78058562,  862.02756831,  855.5322626 ,
        853.92162865,  889.90545757,  921.21167869,  936.2349931 ,
        919.04852675,  929.4960133 ,  929.46372464,  949.52543866,
        995.12132178, 1020.3785741 ,  982.83618011,  998.98942404,
       1012.48116745, 1018.50518642, 1065.36732411, 1109.21877829,
       1057.52166564, 1066.89513105, 1097.13170165, 1098.77346536,
       1136.66084436, 1197.43531965, 1142.81821596, 1138.59785575,
       1179.87686063, 1188.10111119, 1214.4825821 , 1283.37083284,
       1235.06247557, 1218.57716991, 1261.18751067, 1282.0619054 ,
       1302.03408918, 1369.40329154, 1329.79677549, 1308.28050258,
       1344.59858929, 1376.85562624, 1399.08154195, 1459.89591385])

def test_visualization():

    l = [pages, dates, visits]
    _, s = smape_arimaset(5)
    
    vis.fig_detrend(target, l)
    vis.fig_deseasonal(target, l)
    vis.fig_season_trend(target, l)
    vis.fig_forecast(target, l, 550, pred)
    vis.fig_smape_distribution(s)

    assert plt.get_fignums() == 5
