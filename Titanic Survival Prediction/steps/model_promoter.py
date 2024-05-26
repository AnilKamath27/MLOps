from zenml import get_step_context,step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def model_promoter(
    accuracy: float,
    stage: str ="production",
) -> bool:
    """Promotes the model based on the test accuracy.

    Args:
        accuracy: Accuracy of the model.
        stage: Which stage to promote the model to.

    Returns:
        Whether the model was promoted or not.
    """
    logger.info("Promoting the model based on the test accuracy...")

    is_promoted = False

    if accuracy <0.7:
        logger.info(
            f"Model accuracy {accuracy*100:.2f}% is below the threshold of 70! "
            f"Not promoting the model to {stage}."
        )


    else:
        logger.info(f"Model promoted to {stage}!")
        is_promoted = True

        current_model = get_step_context().model

        client = Client()
        try:
            stage_model = client.get_model_version(current_model.name,stage)

            prod_accuracy = (
                stage_model.get_artfiacts("sklearn_classifier")
                .run_metadata["test_accuracy"].value
            )

            if float(accuracy) > float(prod_accuracy):
                is_promoted = False
                current_model.set_stage(stage,force=True)

        except Exception as e:
            is_promoted = True
            current_model.set_stage(stage,force=True)
    return is_promoted