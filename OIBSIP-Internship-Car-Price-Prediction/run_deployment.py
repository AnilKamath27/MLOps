# python run_deployment.py --config deploy
# python run_deployment.py --config predict

import click
from typing import cast
from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Choose to run the deployment pipeline (`deploy`), "
    "run prediction against the deployed model (`predict`), "
    "or both (`deploy_and_predict`). Default is `deploy_and_predict`."
)
@click.option(
    "--min-accuracy",
    default=0.4,
    help="Minimum accuracy required to deploy the model"
)
def run_deployment(config: str, min_accuracy: float):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        continuous_deployment_pipeline(data_path=r"data\car data.csv", min_accuracy=min_accuracy, workers=3, timeout=180)

    if predict:
        inference_pipeline(pipeline_name="continuous_deployment_pipeline", pipeline_step_name="mlflow_model_deployer_step")
        print(f'mlflow ui --backend-store-uri "{get_tracking_uri()}"')

    # Check if there's a running service
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model"
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(f"The MLflow prediction server is running locally as a daemon "
                  f"process service and accepts inference requests at:\n"
                  f"{service.prediction_url}\n"
                  f"To stop the service, run "
                  f"[italic green]`zenml model-deployer models delete "
                  f"{str(service.uuid)}`[/italic green].")
        elif service.is_failed:
            print(f"The MLflow prediction server is in a failed state:\n"
                  f"Last state: '{service.status.state.value}'\n"
                  f"Last error: '{service.status.last_error}'")
    else:
        print("No MLflow prediction server is currently running. The deployment "
              "pipeline must run first to train a model and deploy it. Execute "
              "the same command with the `--config deploy` argument to deploy a model.")

if __name__ == '__main__':
    run_deployment()
