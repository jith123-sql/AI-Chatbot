from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'ANN_Project'

# load the model
model_path = MODEL_DIR / 'chrurn_model.h5'
if not model_path.exists():
    st.error(f"Model file not found: {model_path}")
    st.stop()

model = tf.keras.models.load_model(model_path)


# load all encoders and scaler
with open(MODEL_DIR / 'onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open(MODEL_DIR / 'label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open(MODEL_DIR / 'scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


#streamlit app
st.title("Customer Churn Prediction")

# user_inputs
geography = st.selectbox("Select Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Select Gender", label_encoder_gender.classes_)
age = st.slider("Enter Age", min_value=18, max_value=100, value=30)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])

has_cr_card = 1 if has_cr_card == 'Yes' else 0
is_active_member = 1 if is_active_member == 'Yes' else 0

# prepare the input data
input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Balance': [balance],
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

#one hot encode the geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#combine input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# scale the input data
inputdata_scaled = scaler.transform(input_data)

# prediction_churn
prediction = model.predict(inputdata_scaled, verbose=0)
prediction_prod = prediction[0][0]

if prediction_prod > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")