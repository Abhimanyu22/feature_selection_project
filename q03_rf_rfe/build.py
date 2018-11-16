# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def rf_rfe(data):
    
    X,y = data.iloc[:,:-1], data.iloc[:,-1]
    model = RandomForestClassifier()
    rfe = RFE(model, int(data.shape[1]/2))
    rfe.fit(X,y)
    cols_list = rfe.get_support(indices=True)
    cols_sort = [cols_list for _, cols_list in sorted(zip(rfe.ranking_[cols_list],cols_list))]
    selected_cols = data.iloc[:,cols_sort]

    return list(selected_cols.columns.values)

rf_rfe(data)




