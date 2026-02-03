import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ------------------ Load Model & Encoders ------------------
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ------------------ App Title ------------------
st.title("📉 Customer Churn Prediction App")
st.markdown(
    "Predict whether a customer is likely to **churn** based on their profile and account details."
)
st.divider()

# ------------------ Sidebar Inputs ------------------
st.sidebar.header("🧾 Customer Details")

geography = st.sidebar.selectbox(
    'Geography',
    onehot_encoder_geo.categories_[0]
)

gender = st.sidebar.selectbox(
    'Gender',
    label_encoder_gender.classes_
)

age = st.sidebar.slider('Age', 18, 92, 30)
tenure = st.sidebar.slider('Tenure (Years)', 0, 10, 3)
num_of_products = st.sidebar.slider('Number of Products', 1, 4, 1)

has_cr_card = st.sidebar.selectbox('Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1])

# ------------------ Main Inputs ------------------
st.subheader("💰 Financial Information")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=650)

with col2:
    balance = st.number_input('Account Balance', value=50000.0)

with col3:
    estimated_salary = st.number_input('Estimated Salary', value=60000.0)

st.divider()

# ------------------ Prepare Input Data ------------------
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)

input_data_scaled = scaler.transform(input_data)

# ------------------ Prediction ------------------
st.subheader("🔍 Prediction")

if st.button("Predict Churn 🚀"):
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.markdown("### 📊 Churn Probability")
    st.progress(float(prediction_proba))

    st.write(f"**Probability:** `{prediction_proba:.2f}`")

    if prediction_proba > 0.5:
        st.error("⚠️ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **not likely to churn**.")
