from zenml import pipeline

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=False)
def train_pipeline(data_path:str):
    "Provide the path to the data source"
    df = ingest_data(data_path)
    X_train, X_test,y_train,y_test= clean_data(df)
    model = train_model(X_train, X_test,y_train,y_test)
    R2Score_value, MeanAbsoluteError_value, MeanSquaredError_value = evaluate_model(model, X_test,y_test)