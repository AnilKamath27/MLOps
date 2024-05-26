from typing import Optional
import yaml
from uuid import UUID
from steps import model_evaluator, model_promoter, model_trainer
from pipelines import feature_engineering
from zenml import pipeline
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


# If you have not run feature pieline, then it will run here and make use of the parameters set in the feature_engg.yaml
def load_yaml_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

@pipeline(enable_cache=True)
def training(
    train_dataset_id: Optional[UUID] = None,
    test_dataset_id: Optional[UUID] = None,
    target: Optional[str] = "Survived",
    model_type: Optional[str] = "rf",
) -> None:
    """
    Model training pipeline.

    Args:
        train_dataset_id: The ID of the train dataset.
        test_dataset_id: The ID of the test dataset.
        target: The target column name.
        model_type: The type of model to train.

    Returns:
        The accuracy of the model.
    """
    try:
        logger.info("Starting the model training pipeline...")

        feature_engineering_config = load_yaml_config(r"configs\feature_engineering.yaml")

        if train_dataset_id is None or test_dataset_id is None:
            # Run feature engineering if dataset IDs are not provided
            train_dataset, test_dataset = feature_engineering(
            **feature_engineering_config["parameters"])

        else:
            client = Client()
            train_dataset = client.get_artifact_version(
                name_id_or_prefix=train_dataset_id
            )
            test_dataset = client.get_artifact_version(
                name_id_or_prefix=test_dataset_id
            )

        model = model_trainer(
            train_dataset= train_dataset,
            target= target,
            model_type=model_type,
        )

        acc = model_evaluator(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            target=target,
        )

        model_promoter(accuracy=acc)

        logger.info("Model training pipeline completed successfully.")

    except Exception as e:
        logger.error("Error while training the model: %s", str(e))
        raise e