import os
from typing import Optional

import yaml
from zenml.client import Client
from zenml.logger import get_logger

from pipelines import feature_engineering, training, inference

logger = get_logger(__name__)


def main(
    train_dataset_name: str = "train_dataset",
    train_dataset_version_name: Optional[str] = None,
    test_dataset_name: str = "test_dataset",
    test_dataset_version_name: Optional[str] = None,
    training_pipeline: bool = False,
    feature_pipeline: bool = False,
    inference_pipeline: bool = False,
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

    if inference_pipeline:
        run_args_inference = {}
        pipeline_args = {
            "enable_cache": False,
            "config_path": os.path.join(config_folder, "inference.yaml"),
        }

        # Configure the pipeline
        inference_configured = inference.with_options(**pipeline_args)

        # Fetch the production model
        with open(pipeline_args["config_path"], "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        zenml_model = client.get_model_version(
            config["model"]["name"], config["model"]["version"]
        )
        preprocess_pipeline_artifact = zenml_model.get_artifact("preprocess_pipeline")

        # Use the metadata of feature engineering pipeline artifact
        #  to get the random state and target column
        random_state = preprocess_pipeline_artifact.run_metadata["random_state"].value
        target = preprocess_pipeline_artifact.run_metadata["target"].value
        run_args_inference["random_state"] = random_state
        run_args_inference["target"] = target

        # Run the pipeline
        inference_configured(**run_args_inference)
        logger.info("Inference pipeline finished successfully!")


if __name__ == "__main__":
    # main(feature_pipeline=True)

    # main(
    #     train_dataset_name="train_dataset",
    #     test_dataset_name="test_dataset",
    #     training_pipeline=True,
    #     train_dataset_version_name="44",
    #     test_dataset_version_name="44",
    # )

    # main(
    #     inference_pipeline=True,
        
    # )

    main(
        train_dataset_name="train_dataset",
        test_dataset_name="test_dataset",
        training_pipeline=True,
        inference_pipeline=True,
        train_dataset_version_name="44",
        test_dataset_version_name="44",
    )
