import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicTransformer(BaseEstimator, TransformerMixin):
    """A custom scikit-learn transformer to apply cyclic encoding to periodic variables 
    (i.e., month, longitude)

    These transformers will be applied by sklearn.compose.ColumnTransformer

    Inheriting TransformerMixin gives us fit_transform method for free, 
    and BaseEstimator provides get_params and set_params (for grid searching)

    """
    def __init__(self, col, period, drop_original=True):
        self.col = col # column name to be transformed (i.e. 'month' or 'lon')
        self.period = period # i.e. 12 for month or 360 for lon
        self.drop_original = drop_original # whether or not to drop the un-transformed column

    def fit(self, X, y=None):
        """Mandatory to define transformers -- does nothing here
        """
        return self

    def transform(self, X):
        """Applies the transformation to the columns (X)"""

        X = X.copy()

        # check if the column name exists in the data, otherwise return untransformed data
        if self.col not in X:
            return X

        # convert convert column value into an angle on a circle
        # --> divide the value by the period to get a fraction, then
        # multiplies by 2pi to map it onto a a circle in radians.
        radians = 2 * np.pi * (X[self.col] / self.period)

        # add transformed variables to the dataset
        X[f"{self.col}_sin"] = np.sin(radians)
        X[f"{self.col}_cos"] = np.cos(radians)

        # optionally drop un-transformed variable
        if self.drop_original:
            X = X.drop(columns=[self.col])
        return X

    def get_feature_names_out(self):
        """A numpy array with all feature names transformed or added.

        Note: original feature not included if dropped.
        """
        base = [] if self.drop_original else [self.col]
        return np.array(base + [f"{self.col}_sin", f"{self.col}_cos"])
