from typing import List, Optional
from steps import data_loader, data_splitter, data_preprocessor

from zenml import pipeline, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline(enable_cache=False)
def feature_engineering(
    test_size: float = 0.33,
    drop_duplicates: Optional[bool] = True,
    drop_columns: Optional[List[str]] = None,
    random_state: int = 42,
    normalize: Optional[bool] = True,
    target: Optional[str] = "Survived",
    shuffle: Optional[bool] = True,
    fill_missing: Optional[bool] = True,
):
    """This is an feature engineering pipeline. That does the following things:

    1. Ingests the data.
    2. Splits the data into train, test and inference (if required).
    3. Cleans the data.

    Args:
        test_size: Size of the test set.
        drop_na: Whether to drop rows with na values.
        drop_columns: List of columns to drop.
        random_state: Random state for train test split.
        normalise: Whether to normalise the data.
        Survived: Target column name.

    returns:
        The pre processsed data (dataset_trn,dataset_tst,dataset_inf)"""

    raw_data = data_loader(random_state=random_state, target=target)
    train_dataset, test_dataset = data_splitter(
        raw_data, test_size=test_size, shuffle=shuffle
    )

    train_dataset, test_dataset, _ = data_preprocessor(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        drop_duplicates=drop_duplicates,
        normalize=normalize,
        drop_columns=drop_columns,
        random_state=random_state,
        target=target,
        fill_missing=fill_missing,
    )
