from typing import Union
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from zenml.logger import get_logger

logger = get_logger(__name__)


class DuplicateDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    # @staticmethod
    def transform(self,X):
        return X.drop_duplicates()


class MissingValuesFiller(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    # @staticmethod
    def transform(self,X):
        return X.ffill().bfill()

class ColumnsDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)


class FamilySizeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, sibsp_column: str, parch_column: str):
        self.sibsp_column = sibsp_column
        self.parch_column = parch_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["FamilySize"] = X[self.sibsp_column] + X[self.parch_column] + 1
        return X.drop(columns=[self.sibsp_column, self.parch_column])


class DataFrameCaster(BaseEstimator, TransformerMixin):
    """Support class to cast type back to pd.DataFrame in sklearn Pipeline."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)


class DataEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.get_dummies(X, drop_first=True, dtype=int)
