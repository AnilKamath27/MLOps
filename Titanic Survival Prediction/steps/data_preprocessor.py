from typing import List, Optional, Tuple, Annotated
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from zenml import step, log_artifact_metadata
from utils import (
    DuplicateDropper,
    FamilySizeAdder,
    ColumnsDropper,
    DataFrameCaster,
    MissingValuesFiller,
    DataEncoder,
)

from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def data_preprocessor(
    random_state: int,
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    drop_duplicates: Optional[bool] = True,
    normalize: Optional[bool] = True,
    drop_columns: Optional[List[str]] = True,
    target: Optional[str] = "Survived",
    fill_missing: Optional[bool] = True,
) -> Tuple[
    Annotated[pd.DataFrame, "train_dataset"],
    Annotated[pd.DataFrame, "test_dataset"],
    Annotated[Pipeline, "preprocess_pipeline"],
]:
    """Preprocesses the dataset.

    Args:
        random_state: Random state for train test split.
        train_dataset: The train dataset.
        test_dataset: The test dataset.
        drop_duplicates: Drops all duplicate values.
        normalize: Whether to normalize the data.
        drop_columns: List of columns to drop.
        target: Target column name.

    Returns:
        The preprocessed train and test datasets and the preprocess pipeline.
    """
    logger.info("Preprocessing the dataset...")

    preprocess_pipeline = Pipeline([("passthrough", "passthrough")])

    if drop_duplicates:
        preprocess_pipeline.steps.append(("duplicate_dropper", DuplicateDropper()))

    if fill_missing:
        preprocess_pipeline.steps.append(
            ("missing_values_filler", MissingValuesFiller())
        )

    if drop_columns:
        preprocess_pipeline.steps.append(
            ("columns_dropper", ColumnsDropper(drop_columns))
        )

    preprocess_pipeline.steps.append(
        ("family_size", FamilySizeAdder(sibsp_column="SibSp", parch_column="Parch"))
    )

    preprocess_pipeline.steps.append(("DataEncoder", DataEncoder()))

    if normalize:
        preprocess_pipeline.steps.append(("Scaling", MinMaxScaler()))

    columns = [
        "Survived",
        "Pclass",
        "Age",
        "Fare",
        "FamilySize",
        "Sex",
        "Embarked_Q",
        "Embarked_S",
    ]
    # preprocess_pipeline.steps.append(("cast", DataFrameCaster(train_dataset.columns)))

    preprocess_pipeline.steps.append(("cast", DataFrameCaster(columns)))

    train_dataset_transformed = preprocess_pipeline.fit_transform(train_dataset)
    test_dataset_transformed = preprocess_pipeline.transform(test_dataset)

    log_artifact_metadata(
        artifact_name="preprocess_pipeline",
        metadata={"random_state": random_state, "target": target},
    )
    return train_dataset_transformed, test_dataset_transformed, preprocess_pipeline


# drop_duplicates - Done
# fill_missing - Done
# drop_columns - Done
# Feat engg - Done
# Encoding - Done
# normalize -Done
