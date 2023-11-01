# To run use : streamlit run streamlit_app.py

import streamlit as st
import json
import numpy as np
import pandas as pd
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from pipelines.utils import get_data_for_test

# Load your trained model and deployment service
pipeline_name = "continuous_deployment_pipeline"
pipeline_step_name = "mlflow_model_deployer_step"
model_name = "model"
model_deployer = MLFlowModelDeployer.get_active_model_deployer()
service = model_deployer.find_model_server(
    pipeline_name=pipeline_name,
    pipeline_step_name=pipeline_step_name,
    model_name=model_name,
    running=True)[0]

# Create a Streamlit app
st.title('Car Price Prediction')

# Input form for car details
st.subheader('Enter Car Details:')
year = st.slider('Year', min_value=1990, max_value=2023)
selling_price = st.number_input('Selling Price (in lakhs)', min_value=0.0)
present_price = st.number_input('Present Price (in lakhs)', min_value=0.0)
fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
seller_type = st.selectbox('Seller Type', ['Dealer', 'Individual'])
transmission_type = st.selectbox('Transmission Type', ['Manual', 'Automatic'])
owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])

if st.button('Predict'):
    # Create an input dictionary
    input_data = {
        "data": {
            "Year": year,
            "Selling_Price": selling_price,
            "Present_Price": present_price,
            "price": present_price - selling_price,
            "Driven_kms": 10000,  # Example value
            "Fuel_Type": fuel_type,
            "Selling_type": seller_type,
            "Transmission": transmission_type,
            "Owner": owner
        }
    }

    # Convert input to a JSON string
    input_json = json.dumps(input_data)

    # Make a prediction using the model service
    prediction = service.predict(input_json)

    # Display the prediction
    st.subheader('Predicted Car Price:')
    st.write(f'{prediction[0]:.2f} lakhs')

# To stop the Streamlit app, use the "Stop" button in your terminal or stop the Streamlit server.

# To run the Streamlit app, use the following command in your terminal:
# streamlit run your_file_name.py
