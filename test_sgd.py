from izlearn.linear_model import SGDRegressors
from izlearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import numpy as np
import pandas as pd


X_df_train = pd.read_csv('california_housing_train.csv')
X_df_test = pd.read_csv('california_housing_test.csv')

X_train = X_df_train.drop(columns=['median_house_value'],axis=1)
y_train = X_df_train['median_house_value']

X_test = X_df_test.drop(columns=['median_house_value'],axis=1)
y_test = X_df_test['median_house_value']


# Assuming X_train, y_train, X_test, y_test are defined
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

configs = [
    {"learning_rate": "constant", "eta0": 0.01, "penalty": "l2"},
    {"learning_rate": "constant", "eta0": 0.01, "penalty": "l1"},
    {"learning_rate": "constant", "eta0": 0.01, "penalty": "elasticnet"},
    {"learning_rate": "constant", "eta0": 0.01, "penalty": None},
    {"learning_rate": "invscaling", "eta0": 0.01, "penalty": "l2"},
    {"learning_rate": "optimal", "eta0": 0.01, "penalty": "l2"},
]

for cfg in configs:
    m1 = SGDRegressors(**cfg, max_iter=1000, shuffle=False, random_state=42)
    m2 = SGDRegressor(**cfg, max_iter=1000, shuffle=False, random_state=42)
    
    m1.fit(X_train, y_train)
    m2.fit(X_train, y_train)
    
    print(f"{cfg['learning_rate']} | {cfg['penalty']}")
    print(f"  Train — Mày: {m1.score(X_train, y_train):.4f} | Sklearn: {m2.score(X_train, y_train):.4f}")
    print(f"  Test  — Mày: {m1.score(X_test, y_test):.4f} | Sklearn: {m2.score(X_test, y_test):.4f}")
    print()