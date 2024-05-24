from typing import Annotated, Tuple
import pandas as pd

from sklearn.model_selection import train_test_split

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def data_splitter(dataset:pd.DataFrame,
                  test_size:float= 0.33,
                  random_state:int=42,
                  shuffle:bool=True
)-> Tuple[
    Annotated[pd.DataFrame,"train_dataset"],
    Annotated[pd.DataFrame,"test_dataset"]
]:

    """Splits the dataset into train and test sets.

    Args:
        dataset: The dataset to split.
        test_size: The size of the test set.
        random_state: The random state to use for the split.

    Returns:
        The train and test datasets.
    """
    logger.info("Splitting the dataset into train and test sets...")

    train_dataset, test_dataset = train_test_split(
        dataset, test_size=test_size, random_state=random_state, shuffle=shuffle
    )

    train_dataset = pd.DataFrame(train_dataset, columns=dataset.columns)
    test_dataset = pd.DataFrame(test_dataset, columns=dataset.columns)

    return train_dataset, test_dataset
