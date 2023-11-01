import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from model.model_dev import RandomForestRegressorModel, GradientBoostingRegressorModel, XGBRegressorModel
from steps.config import ModelNameConfig

import mlflow 
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train:pd.Series,
    y_test: pd.Series,
    config:ModelNameConfig)-> RegressorMixin:
    """Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin"""
    
    try:
        model = None
        if config.model_name == "RandomForestRegressor":
            mlflow.sklearn.autolog()
            model = RandomForestRegressorModel()
        elif config.model_name == "GradientBoostingRegressor":
            mlflow.sklearn.autolog()
            model = GradientBoostingRegressorModel()
        elif config.model_name == "XGBRegressorModel":
            mlflow.xgboost.autolog()
            model = XGBRegressorModel()
        else:
            raise ValueError("Model name not supported")
        
        logging.info(f"The {config.model_name} model is being trained")

        trained_model = model.train(X_train, y_train)
        return trained_model
    
    except Exception as e:
        logging.info(f"Error while trainin the model {e}")
        raise e