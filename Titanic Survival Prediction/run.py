import os
import yaml
from pipelines import feature_engineering
from zenml.client import Client
from typing import Optional

from zenml.logger import get_logger

logger = get_logger(__name__)


def main(
    train_dataset_name: str = "train_dataset",
    train_dataset_version_name: Optional[str] = None,
    test_dataset_name: str = "test_dataset",
    test_dataset_version_name: Optional[str] = None,
    feature_pipeline: bool = True,
    training_pipeline: bool = True,
    inference_pipeline: bool = True,
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
            f"artifacts: \n\n1. Train Dataset - Name: {train_dataset_name}, "
            f"Version Name: {train_dataset_artifact.version} \n2. Test Dataset: "
            f"Name: {test_dataset_name}, Version Name: {test_dataset_artifact.version}"
        )


if __name__ == "__main__":
    main()
