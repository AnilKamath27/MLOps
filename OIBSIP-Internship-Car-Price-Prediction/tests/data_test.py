# pytest tests

import logging
import pytest
from steps.ingest_data import IngestData
from steps.clean_data import clean_data
from zenml import step

def test_data_shape():
    """Test the shape of the data after the data cleaning step."""
    try:
        # Ingest data
        ingest_data = IngestData(r"data\car data.csv")
        df = ingest_data.get_data()
        X_train, X_test, y_train, y_test = clean_data(df)

        # Check data shapes
        assert X_train.shape == (201, 7), "The shape of the training set is not correct."
        assert y_train.shape == (201,), "The shape of labels of the training set is not correct."
        assert X_test.shape == (100, 7), "The shape of the testing set is not correct."
        assert y_test.shape == (100,), "The shape of labels of the testing set is not correct."

        logging.info("Data Shape Assertion test passed.")
    except Exception as e:
        pytest.fail(str(e))
    print("test_data_shape executed")

@pytest.fixture
def X_train():  # Define X_train fixture
    ingest_data = IngestData(r"data\car data.csv")
    df = ingest_data.get_data()
    X_train, _, _, _ = clean_data(df)
    return X_train

@pytest.fixture
def X_test():  # Define X_test fixture
    ingest_data = IngestData(r"data\car data.csv")
    df = ingest_data.get_data()
    _, X_test, _, _ = clean_data(df)
    return X_test

def test_check_data_leakage(X_train, X_test):
    """Test if there is any data leakage."""
    try:
        train_set = set(tuple(row) for row in X_train)
        test_set = set(tuple(row) for row in X_test)

        assert not bool(train_set.intersection(test_set)), "There is data leakage."
        logging.info("Data Leakage test passed.")
    except Exception as e:
        pytest.fail(str(e))
    print("check_data_leakage executed")

# pytest tests

