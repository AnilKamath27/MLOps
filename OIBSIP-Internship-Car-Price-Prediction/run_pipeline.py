from pipelines.training_pipeline import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == '__main__':
    train_pipeline(data_path=r"C:\Users\anilk\Desktop\Juypter Notebook\OIBSIP-Internship\OIBSIP-Internship-Car-Price-Prediction\data\car data.csv")

    print(f'mlflow ui --backend-store-uri "{get_tracking_uri()}"')





    # mlflow ui --backend-store-uri "file:C:\Users\anilk\AppData\Roaming\zenml\local_stores\e31e0a6b-eca0-4140-a23e-951c8fc68ef3\mlruns"
    # Stack 'mlflow_stack_car' with id '9824223e-0af1-4dea-b663-669e78c282e5' is owned by user default and is 'private'.
    
    # Commands
    
    # zenml disconnect

    # zenml experiment-tracker flavor list
    
    # python run_pipeline.py
    # zenml up --blocking on windows and zenml up for mac. Password: default

    # zenml stack list
    # zenml stack describe

    # Register your experiment tracker. 
    # zenml experiment-tracker register mlflow_tracker_"name" --flavor=mlflow
    # zenml model-deployer register mlflow_"name" --flavor=mlflow
    # zenml stack register mlflow_stack_"name" -a default -o default -d mlflow_"name" -e mlflow_tracker_"name" --set

    # mlflow ui

#     zenml experiment-tracker register mlflow_tracker_car --flavor=mlflow
#     zenml model-deployer register mlflow_car_price --flavor=mlflow
#     zenml stack register mlflow_stack_car_price -a default -o default -d mlflow_car_price -e mlflow_tracker_car --set

    