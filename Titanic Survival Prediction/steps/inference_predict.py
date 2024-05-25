from typing import Any

import pandas as pd
from typing_extensions import Annotated

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def inference_predict(
    model: Any,
    dataset_inf: pd.DataFrame,
) -> Annotated[pd.DataFrame, "predictions"]:
    """Predictions step.

    This is an example of a predictions step that takes the data and model in
    and returns predicted values.

    This step is parameterized, which allows you to configure the step
    independently of the step code, before running it in a pipeline.
    In this example, the step can be configured to use different input data.
    See the documentation for more information:

        https://docs.zenml.io/user-guide/advanced-guide/configure-steps-pipelines

    Args:
        model: Trained model.
        dataset_inf: The inference dataset.

    Returns:
        The predictions as pandas series
    """
    # run prediction from memory
    predictions = model.predict(dataset_inf)

    predictions = pd.DataFrame(predictions, columns=["predictions"])
    return predictions
