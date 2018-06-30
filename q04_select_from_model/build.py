# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


def select_from_model(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    rfc = RandomForestClassifier(random_state=9)
    rfc.fit(X, y)
    sfm = SelectFromModel(rfc, prefit=True)
    return list(X.columns[sfm.get_support()])


