import pandas as pd
from sklearn.pipeline import Pipeline
from typing_extensions import Annotated

from zenml import step

@step
def inference_preprocessor(
    dataset_inf: pd.DataFrame,
    preprocess_pipeline: Pipeline,
    target: str="Survived",
) -> Annotated[pd.DataFrame, "inference_dataset"]:
    """Data preprocessor step.

    This is an example of a data processor step that prepares the data so that
    it is suitable for model inference. It takes in a dataset as an input step
    artifact and performs any necessary preprocessing steps based on pretrained
    preprocessing pipeline.

    Args:
        dataset_inf: The inference dataset.
        preprocess_pipeline: Pretrained `Pipeline` to process dataset.
        target: Name of target columns in dataset.

    Returns:log_artifact_metadata
        The processed dataframe: dataset_inf.
    """

    # artificially adding `target` column to avoid Pipeline issues
    # dataset_inf[target] = pd.Series([1] * dataset_inf.shape[0])
    dataset_inf = preprocess_pipeline.transform(dataset_inf)
    dataset_inf.drop(columns=[target], inplace=True)
    return dataset_inf

