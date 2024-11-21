import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        """
        Initialize the Winsorizer with the lower and upper quantile thresholds.
        :param lower_quantile: Lower quantile threshold (default is 0.05, i.e., 5%)
        :param upper_quantile: Upper quantile threshold (default is 0.95, i.e., 95%)
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        """
        Compute the quantile thresholds for clipping and save them as instance attributes.
        :param X: Input data (can be a 1D or 2D array or DataFrame)
        :param y: Target values (not used for fitting, included for compatibility with scikit-learn)
        :return: self (the fitted transformer instance)
        """
        X = np.asarray(X)  # Ensure input is a NumPy array
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # Reshape 1D array into 2D array

        # Compute lower and upper quantiles
        self.lower_quantile_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        """
        Clip the data based on the pre-computed quantile thresholds.
        :param X: Input data to be transformed
        :return: Transformed data with values clipped within the quantile range
        """
        # Ensure that the fit method has been called before transforming
        check_is_fitted(self, ["lower_quantile_", "upper_quantile_"])

        X = np.asarray(X)  # Ensure input is a NumPy array
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # Reshape 1D array into 2D array

        # Clip values within the quantile thresholds
        return np.clip(X, self.lower_quantile_, self.upper_quantile_)

    def fit_transform(self, X, y=None):
        """
        Combine the fit and transform steps into a single method.
        :param X: Input data to fit and transform
        :param y: Target values (not used for fitting, included for compatibility with scikit-learn)
        :return: Transformed data
        """
        return self.fit(X, y).transform(X)

