from typing import List, Optional,Tuple
from steps import data_loader, data_splitter, data_preprocessor
import pandas as pd
from zenml import pipeline, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def feature_engineering(
    test_size: float = 0.33,
    drop_duplicates: Optional[bool] = True,
    drop_columns: Optional[List[str]] = None,
    random_state: int = 42,
    normalize: Optional[bool] = True,
    target: Optional[str] = "Survived",
    shuffle: Optional[bool] = True,
    fill_missing: Optional[bool] = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:

    """This is a feature engineering pipeline that performs the following steps:

    1. Ingests the data.
    2. Splits the data into train, test and inference (if required).
    3. Cleans the data.

    Args:
        test_size (float): Size of the test set.
        drop_duplicates (Optional[bool]): Whether to drop duplicate rows.
        drop_columns (Optional[List[str]]): List of columns to drop.
        random_state (int): Random state for train-test split.
        normalize (Optional[bool]): Whether to normalize the data.
        target (Optional[str]): Target column name.
        shuffle (Optional[bool]): Whether to shuffle the data before splitting.
        fill_missing (Optional[bool]): Whether to fill missing values.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]: 
        The preprocessed training, test, and (if applicable) inference datasets.
    """

    raw_data = data_loader(random_state=random_state, target=target)
    train_dataset, test_dataset = data_splitter(
                dataset=raw_data, test_size=test_size, shuffle=shuffle
    )

    train_dataset, test_dataset,_ = data_preprocessor(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        drop_duplicates=drop_duplicates,
        normalize=normalize,
        drop_columns=drop_columns,
        random_state=random_state,
        target=target,
        fill_missing=fill_missing,
    )

    return train_dataset, test_dataset
