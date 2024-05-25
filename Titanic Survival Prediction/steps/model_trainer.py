from typing import Optional, Annotated

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from zenml import ArtifactConfig, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step()
def model_trainer(
        train_dataset: pd.DataFrame,
        model_type: str = "logreg",
        target: Optional[str] = "Survived",
) -> Annotated[
    ClassifierMixin, ArtifactConfig(name="sklearn_classifier", is_model_artifact=True)
]:
    """Trains a model and returns it.

    Args:
        train_dataset: The train dataset.
        target: The target column name.
        model_type: The type of model to train.

    Returns:
        The trained model.
    """
    logger.info("Training the model...")

    if model_type == "logreg":
        model = LogisticRegression(n_jobs=-1, random_state=42)
    elif model_type == "rf":
        model = RandomForestClassifier(n_jobs=-1, random_state=42)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    logger.info(f"Training {model_type} model...")

    model.fit(train_dataset.drop(columns=[target]), train_dataset[target])
    return model
