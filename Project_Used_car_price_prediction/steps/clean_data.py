import logging
import pandas as pd
from zenml import step
from model.data_cleaning import DataCleaning, DataDivideScaleStrategy, DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "Y_train"],
    Annotated[pd.Series, "Y_test"]]:
    '''Clean and preprocess the input DataFrame using two data processing strategies.
    
    Args:
        df (pd.DataFrame): The input DataFrame to be processed.
        
    Returns:
        Tuple[
            Annotated[pd.DataFrame, "X_train"],
            Annotated[pd.DataFrame, "X_test"],
            Annotated[pd.Series, "Y_train"],
            Annotated[pd.Series, "Y_test"]]: Processed data for model training.
    '''
    try:
        data_cleaning = DataCleaning(df, DataPreProcessStrategy())
        processed_data = data_cleaning.handle_data()

        data_cleaning = DataCleaning(processed_data, DataDivideScaleStrategy())
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed")

        return X_train, X_test, y_train, y_test  #type: ignore

    except Exception as e:
        logging.info(f"Error in cleaning data {e}")
        raise e
