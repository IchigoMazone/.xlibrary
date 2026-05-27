from sklearn.preprocessing import StandardScaler
import numpy as np

X_train_first = np.array([[1, 2], [3, 4], [5, 6]])
X_train_second = np.array([[7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
X_train_end = np.array([[7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])


scaler = StandardScaler()
scaler.partial_fit(X_train_first)
scaler.partial_fit(X_train_second)
scaler.partial_fit(X_train_end)

print(scaler.mean_)
print(scaler.scale_)
print(scaler.var_)
print(scaler.n_features_in_)
print(scaler.n_samples_seen_)