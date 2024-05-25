import os
from typing import Optional

from zenml.client import Client
from zenml.logger import get_logger

from pipelines import feature_engineering, training

logger = get_logger(__name__)


def main(
    train_dataset_name: str = "train_dataset",
    train_dataset_version_name: Optional[str] = None,
    test_dataset_name: str = "test_dataset",
    test_dataset_version_name: Optional[str] = None,
    training_pipeline: bool = False,
    feature_pipeline: bool = False,
    # inference_pipeline: bool = False,
    no_cache: bool = False,
):
    client = Client()

    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )

    # Execute Feature Engineering Pipeline
    if feature_pipeline:
        pipeline_args = {}
        if no_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = os.path.join(
            config_folder, "feature_engineering.yaml"
        )
        run_args_feature = {}
        feature_engineering.with_options(**pipeline_args)(**run_args_feature)
        logger.info("Feature Engineering pipeline finished successfully!\n")

        train_dataset_artifact = client.get_artifact_version(train_dataset_name)
        test_dataset_artifact = client.get_artifact_version(test_dataset_name)
        logger.info(
            "The latest feature engineering pipeline produced the following "
            f"artifacts: \n\n1. Train Dataset Name: {train_dataset_name}, "
            f"Version Name: {train_dataset_artifact.version} \n2. Test Dataset: "
            f"Name: {test_dataset_name}, Version Name: {test_dataset_artifact.version}"
        )

    # Execute Training Pipeline
    if training_pipeline:
        run_args_train = {}

        # If train_dataset_version_name is specified, use versioned artifacts
        if train_dataset_version_name or test_dataset_version_name:
            # However, both train and test dataset versions must be specified
            assert (
                train_dataset_version_name is not None
                and test_dataset_version_name is not None
            )
            train_dataset_artifact_version = client.get_artifact_version(
                train_dataset_name, train_dataset_version_name
            )
            # If train dataset is specified, test dataset must be specified
            test_dataset_artifact_version = client.get_artifact_version(
                test_dataset_name, test_dataset_version_name
            )
            # Use versioned artifacts
            run_args_train["train_dataset_id"] = train_dataset_artifact_version.id
            run_args_train["test_dataset_id"] = test_dataset_artifact_version.id

        # Run the SGD pipeline
        pipeline_args = {}
        if no_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = os.path.join(
            config_folder, "training_logreg.yaml"
        )
        training.with_options(**pipeline_args)(**run_args_train)
        logger.info(
            "Training pipeline with Logistic Regression finished successfully!\n\n"
        )

        # Run the RF pipeline
        pipeline_args = {}
        if no_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = os.path.join(config_folder, "training_rf.yaml")
        training.with_options(**pipeline_args)(**run_args_train)
        logger.info("Training pipeline with RF finished successfully!\n\n")


if __name__ == "__main__":
    main(training_pipeline=True)
    # main(train_dataset_name="train_dataset",
    #      test_dataset_name="test_dataset",
    #      training_pipeline=True,
    #      train_dataset_version_name="40",
    #      test_dataset_version_name="40")


# 1. Train Dataset - Name: train_dataset, Version Name: 40
# 2. Test Dataset: Name: test_dataset, Version Name: 40
