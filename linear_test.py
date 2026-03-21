from izlearn.linear_mode import LinearRegression
from izlearn.preprocessing import StandardScaler
from izlearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

X_df_train = pd.read_csv('california_housing_train.csv')
X_df_test = pd.read_csv('california_housing_test.csv')

X_train = X_df_train.drop(columns=['median_house_value'],axis=1)
y_train = X_df_train['median_house_value']

X_test = X_df_test.drop(columns=['median_house_value'],axis=1)
y_test = X_df_test['median_house_value']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print('R2 score:', r2_score(y_test, y_pred))
print('R2 score:', r2_score(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))



