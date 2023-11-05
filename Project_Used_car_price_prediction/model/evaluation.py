import logging 
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class Evaluation(ABC):
    """Abstract class defining strategy for evaluation of model"""
    @abstractmethod
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """Calculates the scores for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            None """
        pass

class R2Score(Evaluation):
    """Evaluation strategy that calculates the R^2 score (Coefficient of Determination)"""
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R^2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R^2 Score of the model: {r2}")
            return r2
        except Exception as e:
            logging.info(f"Error while calculating R^2 Score: {e}")
            raise e

class MeanAbsoluteError(Evaluation):
    """Evaluation strategy that calculates the Mean Absolute Error (MAE)"""
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Mean Absolute Error")
            mae = mean_absolute_error(y_true, y_pred)
            logging.info(f"Mean Absolute Error of the model: {mae}")
            return mae
        except Exception as e:
            logging.info(f"Error while calculating Mean Absolute Error: {e}")
            raise e

class MeanSquaredError(Evaluation):
    """Evaluation strategy that calculates the Mean Squared Error (MSE)"""
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Mean Squared Error")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Mean Squared Error of the model: {mse}")
            return mse
        except Exception as e:
            logging.info(f"Error while calculating Mean Squared Error: {e}")
            raise e