import logging
import pandas as pd
from zenml import step

class IngestData():
    ''' Ingest the data source path and returns the pandas DataFrame '''
    def __init__(self, data_path):
        ''' Takes in data source path
        Args: 
            Data source path '''
        self.data_path = data_path

    def get_data(self):
        ''' Ingesting the data from the data path & returns the DataFrame '''
        logging.info(f"Ingesting the data source path {self.data_path}")
        return pd.read_csv(self.data_path) 
    
@step
def ingest_data(data_path:str)-> pd.DataFrame:
    ''' Ingests the data source path and returns the DataFrame
    Args: 
        Data source path
    Returns:
        pd.DataFrame: The ingested data '''
    try:
        ingest_df = IngestData(data_path)
        return ingest_df.get_data()
    
    except Exception as e:
        logging.error(f"Error fetching data from {data_path}")
        raise e
