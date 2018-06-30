# %load q05_forward_selected/build.py
# Default imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()

def forward_selected(data, model):
    variables = []
    r2_scores = []
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    current_df = pd.DataFrame()
    for i in range(len(X.columns)):
        max_r2_score = 0
        selected_col = ''
        for j in X.columns:
            if j not in variables:
                input_df = current_df.copy()
                input_df[j] = X[j]
                model.fit(input_df, y)
                y_pred = model.predict(input_df)
                current_r2_score = r2_score(y, y_pred)
                if max_r2_score < current_r2_score:
                    max_r2_score = current_r2_score
                    selected_col = j
        current_df[selected_col] = X[selected_col]
        variables.append(selected_col)
        r2_scores.append(max_r2_score)
    return variables, r2_scores
            

