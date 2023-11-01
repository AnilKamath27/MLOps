from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model Configurations"""
    model_name:str = "RandomForestRegressor"

    # Replace RandomForestRegressor with the model name of your choice to run model. 
