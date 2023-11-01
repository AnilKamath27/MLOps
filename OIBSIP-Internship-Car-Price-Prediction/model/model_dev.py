import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.base import RegressorMixin

class Model(ABC):
    '''Abstract class for all models'''
    @abstractmethod
    def train(self, X_train, y_train, **kwargs) -> RegressorMixin:
        """Trains the model
        Args:
            X_train: Training data
            y_train: Training label
        Returns:
            None"""
        
class RandomForestRegressorModel(Model):
    """Random Forest Regressor Model"""
    def train(self, X_train, y_train, **kwargs):
        """Trains the model
        Args:
            X_train: Training data
            y_train: Training label
        Returns:
            None"""     
        try:
            rf = RandomForestRegressor(random_state=42, **kwargs)
            rf.fit(X_train, y_train)
            logging.info("Model training is completed")
            return rf
        
        except Exception as e:
            logging.info(f"Error in training model {e}")
            raise e

class GradientBoostingRegressorModel(Model):
    """Gradient Boosting Regressor Model"""
    def train(self, X_train, y_train, **kwargs):
        """Trains the model
        Args:
            X_train: Training data
            y_train: Training label
        Returns:
            None"""     
        try:
            gb = GradientBoostingRegressor(random_state=42,**kwargs)
            gb.fit(X_train, y_train)
            logging.info("Model training is completed")
            return gb
        
        except Exception as e:
            logging.info(f"Error in training model {e}")
            raise e
        
class XGBRegressorModel(Model):
    """XG Boost Regression Model"""
    def train(self, X_train, y_train, **kwargs):
        """Trains the model
        Args:
            X_train: Training data
            y_train: Training label
        Returns:
            None"""     
        try:
            xg = XGBRegressor(random_state=42,**kwargs)
            xg.fit(X_train, y_train)
            logging.info("Model training is completed")
            return xg
        
        except Exception as e:
            logging.info(f"Error in training model {e}")
            raise e