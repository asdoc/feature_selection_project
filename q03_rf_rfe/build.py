# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


def rf_rfe(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    rfc = RandomForestClassifier()
    number_of_features = df.columns.size//2
    rfe = RFE(rfc, number_of_features)
    rfe.fit(X, y)
    return list(X.columns[rfe.get_support()])


