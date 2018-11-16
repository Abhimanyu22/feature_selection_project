# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

def select_from_model(data):
    X,y = data.iloc[:,:-1], data.iloc[:,-1]
    model = RandomForestClassifier(random_state=9)
    model.fit(X,y)
    rfe = SelectFromModel(model, prefit=True)
    
    rfe.transform(X)
    cols_list = rfe.get_support(indices=True)
    features_selected = data.iloc[:,cols_list]
    
    return list(features_selected.columns.values)

select_from_model(data)
    
    



