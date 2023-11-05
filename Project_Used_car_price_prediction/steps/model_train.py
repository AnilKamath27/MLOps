import logging
import mlflow
import pandas as pd
from model.model_dev import HyperparameterTuner, GradientBoostingRegressorModel,LinearRegressionModel,RandomForestRegressorModel,XGBoostRegressorModel
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client
from .config import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin"""
    try:
        model = None
        tuner = None

        if config.model_name == "GradientBoostingRegressorModel":
            mlflow.sklearn.autolog()
            model = GradientBoostingRegressorModel()
        elif config.model_name == "RandomForestRegressorModel":
            mlflow.sklearn.autolog()
            model = RandomForestRegressorModel()
        elif config.model_name == "XGBoostRegressorModel":
            mlflow.xgboost.autolog()
            model = XGBoostRegressorModel()
        elif config.model_name == "LinearRegressionModel":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
        else:
            raise ValueError("Model name not supported")

        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)

        if config.fine_tuning:
            best_params = tuner.optimize()
            trained_model = model.train(x_train, y_train, **best_params)
        else:
            trained_model = model.train(x_train, y_train)
        return trained_model
    
    except Exception as e:
        logging.error(e)
        raise e