from izlearn.preprocessing import StandardScaler as StandardScalerX
from sklearn.preprocessing import StandardScaler as StandardScalerY
from izlearn.preprocessing import MinMaxScaler as MinMaxScalerZ
from sklearn.preprocessing import MinMaxScaler as MinMaxScalerT
import numpy as np

X_train_first = np.array([[1, 2], [3, 4], [5, 6]])
X_train_second = np.array([[7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
X_train_end = np.array([[7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
X_train = np.vstack((X_train_first, X_train_second, X_train_end))

scalerX = StandardScalerX()
scalerX.partial_fit(X_train_first)
scalerX.partial_fit(X_train_second)
scalerX.partial_fit(X_train_end)

scalerY = StandardScalerY()
scalerY.partial_fit(X_train_first)
scalerY.partial_fit(X_train_second)
scalerY.partial_fit(X_train_end)

X_train_scaled_X = scalerX.transform(X_train)
X_train_scaled_Y = scalerY.transform(X_train)

scalerZ = MinMaxScalerZ(feature_range=(-5, 5))
scalerZ.partial_fit(X_train_first)
scalerZ.partial_fit(X_train_second)
scalerZ.partial_fit(X_train_end)

scalerT = MinMaxScalerT(feature_range=(-5, 5))
scalerT.partial_fit(X_train_first)
scalerT.partial_fit(X_train_second)
scalerT.partial_fit(X_train_end)

X_train_scaled_Z = scalerZ.transform(X_train)
X_train_scaled_T = scalerT.transform(X_train)

print(scalerX.mean_ - scalerY.mean_)
print(scalerX.scale_ - scalerY.scale_)
print(scalerX.var_ - scalerY.var_)
print(scalerX.n_features_in_ - scalerY.n_features_in_)
print(scalerX.n_samples_seen_ - scalerY.n_samples_seen_)
print(X_train_scaled_X - X_train_scaled_Y)

print(scalerT.scale_ - scalerZ.scale_)
print(scalerT.min_ - scalerZ.min_)
print(np.array(scalerZ.feature_range) - np.array(scalerT.feature_range))
print(scalerT.data_range_ - scalerZ.data_range_)
print(scalerT.data_min_ - scalerZ.data_min_)
print(scalerT.data_max_ - scalerZ.data_max_)
print(scalerT.n_features_in_ - scalerZ.n_features_in_)
print(scalerT.n_samples_seen_ - scalerZ.n_samples_seen_)
print(X_train_scaled_T - X_train_scaled_Z)
print(scalerZ.scale_ - scalerT.scale_)
print(scalerZ.n_features_in_ - scalerT.n_features_in_)
print(scalerZ.n_samples_seen_ - scalerT.n_samples_seen_)
