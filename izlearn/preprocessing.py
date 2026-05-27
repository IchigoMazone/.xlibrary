import numpy as np

class StandardScaler:
  def __init__(self):
    self.n_samples_seen_ = None
    self.n_features_in_ = None
    self.mean_ = None
    self.scale_ = None
    self.var_ = None

  def fit(self, X):
    X = np.array(X, dtype=float)
    self.n_samples_seen_ = X.shape[0]
    self.n_features_in_ = X.shape[1]
    self.mean_ = X.mean(axis=0)
    self.scale_ = X.std(axis=0)
    self.var_ = X.var(axis=0)
    return self

  def transform(self, X):
    X = np.array(X, dtype=float)
    return (X - self.mean_) / (self.scale_ + 1e-8)

  def inverse_transform(self, Z):
    Z = np.array(Z, dtype=float)
    return np.round(Z * self.scale_ + self.mean_, 4)

  def fit_transform(self, X):
    X = np.array(X, dtype=float)
    self.fit(X)
    return self.transform(X)

  def partial_fit(self, X):
    X = np.array(X, dtype=float)
    new_n_samples_seen_ = X.shape[0]
    new_n_features_in_ = X.shape[1]
    new_mean_ = X.mean(axis=0)
    new_scale_ = X.std(axis=0)
    new_var_ = X.var(axis=0)

    if self.n_features_in_ != new_n_features_in_ and self.n_features_in_ is not None:
      raise ValueError('X out features')

    if self.n_samples_seen_ is None:
      self.n_samples_seen_ = new_n_samples_seen_
      self.n_features_in_ = new_n_features_in_
      self.mean_ = new_mean_
      self.scale_ = new_scale_
      self.var_ = new_var_
      return self

    old_n_samples_seen_ = self.n_samples_seen_
    old_mean_ = self.mean_
    old_var_ = self.var_

    update_n_samples_seen_ = new_n_samples_seen_ + old_n_samples_seen_
    update_mean_ = (new_n_samples_seen_ * new_mean_ + old_n_samples_seen_ * old_mean_) / update_n_samples_seen_
    total_samples = old_n_samples_seen_ + new_n_samples_seen_
    delta = new_mean_ - old_mean_
    update_var_ = (
        new_n_samples_seen_ * new_var_ + 
        old_n_samples_seen_ * old_var_ + 
        (old_n_samples_seen_ * new_n_samples_seen_ / total_samples) * delta**2
    ) / total_samples
    update_scale_ = np.sqrt(update_var_)

    self.n_samples_seen_ = update_n_samples_seen_
    self.mean_ = update_mean_
    self.scale_ = update_scale_
    self.var_ = update_var_
    return self
  

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
      self.n_samples_seen_ = None
      self.n_features_in_ = None
      self.min_ = None
      self.data_min_ = None
      self.data_max_ = None 
      self.scale_ = None
      self.data_range_ = None
      self.feature_range = feature_range

    def fit(self, X):
      X = np.array(X, dtype=float)
      self.min_ = X.min(axis=0)
      self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (self.data_max_ - self.data_min_ + 1e-8)
      self.n_samples_seen_ = X.shape[0]
      self.n_features_in_ = X.shape[1]
      self.data_min_ = X.min(axis=0)
      self.data_max_ = X.max(axis=0)
      return self

    def transform(self, X):
      X = np.array(X, dtype=float)
      return ((X - self.data_min_) / (self.data_max_ - self.data_min_ + 1e-8)) * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def fit_transform(self, X):
      X = np.array(X, dtype=float)
      self.fit(X)
      return self.transform(X)
    
    def inverse_transform(self, Z):
      Z = np.array(Z, dtype=float)
      return np.round(Z * (self.data_max_ - self.data_min_) + self.data_min_, 4)
    
    def partial_fit(self, X):
      X = np.array(X, dtype=float)
      new_data_min_ = X.min(axis=0)
      new_data_max_ = X.max(axis=0)
      new_data_range_ = new_data_max_ - new_data_min_
      new_n_samples_seen_ = X.shape[0]
      new_n_features_in_ = X.shape[1]
      new_scale_ = (self.feature_range[1] - self.feature_range[0]) / (new_data_max_ - new_data_min_ + 1e-8)

      if self.n_features_in_ != new_n_features_in_ and self.n_features_in_ is not None:
        raise ValueError('X out features')

      if self.n_samples_seen_ is None:
        self.min_ = self.feature_range[0] - new_data_min_ * new_scale_
        self.data_max_ = new_data_max_
        self.data_min_ = new_data_min_
        self.data_range_ = new_data_range_
        self.scale_ = new_scale_
        self.n_samples_seen_ = new_n_samples_seen_
        self.n_features_in_ = new_n_features_in_
        return self
      
      old_n_samples_seen_ = self.n_samples_seen_
      old_data_min_ = self.data_min_
      old_data_max_ = self.data_max_

      update_n_samples_seen_ = new_n_samples_seen_ + old_n_samples_seen_
      update_data_min_ = np.minimum(new_data_min_, old_data_min_)
      update_data_max_ = np.maximum(new_data_max_, old_data_max_)
      update_data_range_ = update_data_max_ - update_data_min_
      update_scale_ = (self.feature_range[1] - self.feature_range[0]) / (update_data_max_ - update_data_min_ + 1e-8)
      update_min_ = self.feature_range[0] - update_data_min_ * update_scale_
      
      self.n_samples_seen_ = update_n_samples_seen_
      self.data_min_ = update_data_min_
      self.data_max_ = update_data_max_
      self.data_range_ = update_data_range_
      self.scale_ = update_scale_
      self.min_ = update_min_
      
      return self
    

