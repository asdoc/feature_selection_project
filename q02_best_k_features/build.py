# %load q02_best_k_features/build.py
# Default imports

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


def percentile_k_features(df, k=20):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    select_percentile = SelectPercentile(f_regression, k)
    select_percentile.fit(X,y)
    dataframe = pd.DataFrame()
    dataframe['cols'] = X.columns
    dataframe['selected'] = select_percentile.get_support()
    dataframe['score'] = select_percentile.scores_
    dataframe.sort_values('score', inplace=True, ascending=False)
    return list(dataframe[dataframe['selected'] == True]['cols'])



