import logging
import pandas as pd
import numpy as np
from zenml import step
from model.evaluation import R2Score, MeanAbsoluteError,MeanSquaredError
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated

import mlflow
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test:  pd.Series)-> Tuple[
                       Annotated[float, "R2Score"],
                       Annotated[float, "MeanAbsoluteError"],
                       Annotated[float, "MeanSquaredError"]]:
    '''Evaluates the model on the chosen metrics.
    Args:
        model (RegressorMixin): The trained classifier model.
        X_test (pd.DataFrame): Test data for evaluation.
        y_test (pd.Series): True labels for the test data.
    Returns:
        Tuple[float, float, float]: A tuple containing R2Score, MeanAbsoluteError, and MeanSquaredError.'''
    try:
        predictions = model.predict(X_test)  # type: ignore

        # Calculate and log R2Score, MeanAbsoluteError, and MeanSquaredError
        r2_class = R2Score()
        r2score = r2_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("R2Score", r2score)

        mae_class = MeanAbsoluteError()
        meanabsoluteerror = mae_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("MeanAbsoluteError", meanabsoluteerror)

        mse_class = MeanSquaredError()
        meansquarederror = mse_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("MeanSquaredError", meansquarederror)

        return (r2score, meanabsoluteerror, meansquarederror)
    
    except Exception as e:
        logging.info(f"Error while evaluating the model {e}")
        raise e
