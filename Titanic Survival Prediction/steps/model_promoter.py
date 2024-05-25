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


# from zenml import get_step_context, step
# from zenml.client import Client
# from zenml.logger import get_logger

# logger = get_logger(__name__)


# @step
# def model_promoter(accuracy: float, stage: str = "production") -> bool:
#     """Model promoter step.

#     This is an example of a step that conditionally promotes a model. It takes
#     in the accuracy of the model and the stage to promote the model to. If the
#     accuracy is below 80%, the model is not promoted. If it is above 80%, the
#     model is promoted to the stage indicated in the parameters. If there is
#     already a model in the indicated stage, the model with the higher accuracy
#     is promoted.

#     Args:
#         accuracy: Accuracy of the model.
#         stage: Which stage to promote the model to.

#     Returns:
#         Whether the model was promoted or not.
#     """
#     is_promoted = False

#     if accuracy < 0.8:
#         logger.info(
#             f"Model accuracy {accuracy*100:.2f}% is below 80% ! Not promoting model."
#         )
#     else:
#         logger.info(f"Model promoted to {stage}!")
#         is_promoted = True

#         # Get the model in the current context
#         current_model = get_step_context().model

#         # Get the model that is in the production stage
#         client = Client()
#         try:
#             stage_model = client.get_model_version(current_model.name, stage)
#             # We compare their metrics
#             prod_accuracy = (
#                 stage_model.get_artifact("sklearn_classifier")
#                 .run_metadata["test_accuracy"]
#                 .value
#             )
#             if float(accuracy) > float(prod_accuracy):
#                 # If current model has better metrics, we promote it
#                 is_promoted = True
#                 current_model.set_stage(stage, force=True)
#         except KeyError:
#             # If no such model exists, current one is promoted
#             is_promoted = True
#             current_model.set_stage(stage, force=True)
#     return is_promoted
