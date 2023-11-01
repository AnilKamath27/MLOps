import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data:pd.DataFrame)->Union[pd.DataFrame, Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]]:
        """Abstract method for handling data according to a specific strategy.
        Args:
            data (pd.DataFrame): Input data to be processed.
        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]: Processed data."""
        pass

class DataPreProcessStrategy(DataStrategy):
    """Strategy for Data Pre Processing"""
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data by dropping columns and encoding categorical features.
        Args:
            data (pd.DataFrame): Input data to be processed.
        Returns:
            pd.DataFrame: Processed data."""
        try:
            cols_to_drop = ["Car_Name"]
            data = data.drop(cols_to_drop, axis=1)
            cat_cols = data.select_dtypes(include=['object'])
            le = LabelEncoder()
            for i in cat_cols:
                data[i] = le.fit_transform(data[i])
                
            data.drop_duplicates(inplace=True) # There are duplicates in the datset.
            return data
        
        except Exception as e:
            logging.info(f"Error in preprocessing the data {e}")
            raise e

class DataDivideScaleStrategy(DataStrategy):
    """Strategy for dividing and scaling data into training and testing sets."""
    def handle_data(self, data: pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
        """Divide and scale data into training and testing sets.
        Args:
            data (pd.DataFrame): Input data to be processed.
        Returns:
            Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]: Processed data."""
        try:
            X = data.drop(["Selling_Price"], axis=1)
            y = data["Selling_Price"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            ss = StandardScaler()
            X_train = ss.fit_transform(X_train)
            X_test = ss.transform(X_test)

            X_train = pd.DataFrame(X_train, columns=X.columns)   
            X_test = pd.DataFrame(X_test, columns=X.columns) 

            return X_train, X_test, y_train, y_test # type: ignore
            # Converting y_train and y_test to numpy because during autolog, it fails to prcoess an pd.Series. 
        
        except Exception as e:
            logging.info(f"Error while splitting and scaling dataset {e}")
            raise e

class DataCleaning:
    def __init__(self, data, strategy: DataStrategy): 
        """Initialize the DataCleaning instance.
        Args:
            data (Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]): Input data.
            strategy (DataStrategy): Data processing strategy."""
        self.data = data
        self.strategy = strategy

    def handle_data(self)-> Union[pd.DataFrame, Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]]:
        """Handle the input data using the specified strategy.
        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]]: Processed data."""
        try:
            return self.strategy.handle_data(self.data)
        
        except Exception as e:
            logging.info(f"Error in handling data {e}")
            raise e
        

# if __name__ == "__main__":
#     df = pd.read_csv(r"data\car data.csv")
#     data_cleaning = DataCleaning(data= df,strategy=DataPreProcessStrategy())
#     processed_data = data_cleaning.handle_data()

#     print(df.shape)

#     data_cleaning = DataCleaning(data=processed_data, strategy=DataDivideScaleStrategy())
#     processed_data = data_cleaning.handle_data()
#     X_train, X_test, y_train, y_test = processed_data

#     print(X_train.shape, y_train.shape)