from pathlib import Path

import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

# load the model
model = tf.keras.models.load_model('chrurn_model.h5')


# load all encoders and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
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

# prepare the input data
input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Balance': balance,
    'Tenure': tenure,
    'NumOfProducts': num_of_products,
    'HasCrCard': 1 if has_cr_card == 'Yes' else 0,
    'IsActiveMember': 1 if is_active_member == 'Yes' else 0,
    'EstimatedSalary': estimated_salary
}

#one hot encode the geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#combine input data
input_df = pd.DataFrame([input_data])
input_df = pd.concat([input_df, geo_encoded_df], axis=1)

# IMPORTANT:
# Use exactly the same feature order used during scaler.fit()
input_df = input_df[scaler.feature_names_in_]

# scale the input data
inputdata_scaled = scaler.transform(input_df)

# prediction_churn
prediction = model.predict(inputdata_scaled, verbose=0)
prediction_prod = prediction[0][0]

if prediction_prod > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")