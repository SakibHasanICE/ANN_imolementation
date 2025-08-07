import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
import tensorflow as tf
import pickle
# Load the trained model


try:
    model = tf.keras.models.load_model('churn_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)


with open('scalar.pkl', 'rb') as f:
    scalar = pickle.load(f)

with open('le.pkl', 'rb') as f:
    le = pickle.load(f)

with open('ohe_ndarray.pkl', 'rb') as f:
    ohe_ndarray = pickle.load(f) 

# Streamlit app title
st.title('Customer Churn Prediction')
# User input
geography = st.selectbox('Geography', ohe_ndarray.categories_[0])
gender = st.selectbox('Gender', le.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [le.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})
 

 # One-hot encode 'Geography'
geo_encoded = ohe_ndarray.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_ndarray.get_feature_names_out(['geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.drop("geography",axis=1), geo_encoded_df], axis=1)
 #Scale the input data
input_data_scaled = scalar.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
