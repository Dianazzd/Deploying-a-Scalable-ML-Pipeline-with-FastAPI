import pytest
import pandas as pd
import numpy as np
import sklearn
import os
from sklearn.model_selection import train_test_split
from ml.model import train_model
from ml.data import process_data


@pytest.fixture
def data():
    data_path = os.path.join(os.getcwd(), "data", "census.csv")
    df = pd.read_csv(data_path)
    return df


def test_train_model(data):
    """
    # check the function uses a RandomForest Classifier
    """
    train, test = train_test_split(data, test_size=0.2)
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
    train, 
    categorical_features=cat_features, 
    label="salary", 
    training=True
    )
    model = train_model(X_train, y_train)
    assert type(model) == sklearn.ensemble._forest.RandomForestClassifier


# TODO: implement the second test. Change the function name and input as needed
def test_dataset_cols(data):
    """
    # check number of columns in the dataset
    """
    assert data.shape[1] == 15
    


# TODO: implement the third test. Change the function name and input as needed
def test_dataset_shape(data):
    """
    # make sure the dataset has no null values
    """
    assert data.shape == data.dropna().shape
    
