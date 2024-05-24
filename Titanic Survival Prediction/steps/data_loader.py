import pandas as pd
from typing import Annotated

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step()
def data_loader(
    random_state: int = 42, is_inference: bool = False, target: str = "Survived"
) -> Annotated[pd.DataFrame, "dataset"]:
    """Ingests the Titanic dataset and returns the dataset as a DataFrame.

    Args:
        random_state: Random state for train-test split.
        is_inference: If 'True', a subset of the data will be used for inference.
        target: Target column name.

    Returns:
        The dataset as a DataFrame.
    """

    try:
        logger.info("Ingesting the data...")
        dataset: pd.DataFrame = pd.read_csv(r"data\titanic.csv")
        inference_size = int(len(dataset) * 0.05)
        inference_subset = dataset.sample(inference_size, random_state=random_state)
        if is_inference:
            dataset = inference_subset
            dataset.drop(columns=target, inplace=True)
        else:
            dataset.drop(inference_subset.index, inplace=True)
        dataset.reset_index(drop=True, inplace=True)
        logger.info(f"Dataset with {len(dataset)} records loaded!")
        return dataset

    except Exception as e:
        logger.error("Error while ingesting the data.")
        raise e


# if __name__ == "__main__":
#     a = data_loader(is_inference=False)
#     print(a)
