# streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import pickle  

st.title('Car Price Prediction')

st.subheader('Enter Car Details:')
year = st.slider('Year', min_value=1990, max_value=2023)
present_price = st.number_input('Present Price (in lakhs)', min_value=0.0)
Driven_kms = st.number_input('Enter the kilometers (kms) driven ', min_value=0.0)
fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
seller_type = st.selectbox('Seller Type', ['Dealer', 'Individual'])
transmission_type = st.selectbox('Transmission Type', ['Manual', 'Automatic'])
owner = st.selectbox('Owner', [0, 1, 3])

if st.button('Predict'):
    with open(r"saved_model\xgboost_model.pkl",mode='rb') as model_file:
        model = pickle.load(model_file)

    input_data = pd.DataFrame({
        'Year': [year],
        'Present_Price': [present_price],
        'Driven_kms': [Driven_kms],
        'Fuel_Type': [fuel_type],
        'Selling_type': [seller_type],
        'Transmission': [transmission_type],
        'Owner': [owner]})

    prediction = model.predict(input_data)

    st.subheader('Predicted Car Price:')
    st.write(f'{prediction[0]:.2f} lakhs')
