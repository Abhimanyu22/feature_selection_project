# %load q02_best_k_features/build.py
# Default imports
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

data = pd.read_csv('data/house_prices_multivariate.csv')

def percentile_k_features(data, k=20):
    
    X = data.iloc[:,:-1]
    y = data['SalePrice']
    f, _ = f_regression(X,y)
    f = list(f)
    xc = list(X.columns)
    
    f_sort = sorted(f)
    xc_s = [x for _,x in sorted(zip(f, xc))]
    req_len = int((k/100)*len(xc_s)) + 1
 
    return xc_s[::-1][:req_len]

    
percentile_k_features(data)
    
    






    




