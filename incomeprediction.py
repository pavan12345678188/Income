import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import pickle

# Load model, scaler, encoder, and encoded columns
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("encoded_columns.pkl", "rb") as f:
    encoded_columns = pickle.load(f)

# Page layout
st.set_page_config(page_title="Income Prediction App", layout="centered")
st.markdown(
    """
    <style>
        .main { background-color: #f0f2f6; padding: 2rem; border-radius: 10px; }
        .stButton>button {
            background-color: #4CAF50; color: white; font-size: 16px; padding: 10px;
        }
        .result {
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💼 Income Prediction App")
st.subheader("Predict whether income is >50K or <=50K based on your profile")

# User input form
def get_user_input():
    with st.form("prediction_form"):
        age = st.slider("Age", 17, 90, 30)
        workclass = st.selectbox("Workclass", [
            'Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov'
        ])
        education = st.selectbox("Education", [
            'HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-acdm', '11th'
        ])
        marital_status = st.selectbox("Marital Status", [
            'Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed'
        ])
        occupation = st.selectbox("Occupation", [
            'Exec-managerial', 'Craft-repair', 'Prof-specialty', 'Adm-clerical', 'Sales', 'Other-service'
        ])
        relationship = st.selectbox("Relationship", [
            'Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife'
        ])
        race = st.selectbox("Race", [
            'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'
        ])
        sex = st.selectbox("Sex", ['Male', 'Female'])
        hours_per_week = st.slider("Hours per Week", 1, 99, 40)
        native_country = st.selectbox("Native Country", [
            'United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India'
        ])
        submitted = st.form_submit_button("Predict Income")

        data = {
            "age": age,
            "workclass": workclass,
            "education": education,
            "marital-status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "hours-per-week": hours_per_week,
            "native-country": native_country
        }

        return pd.DataFrame([data]), submitted

# Get user input
input_df, submitted = get_user_input()

if submitted:
    # One-hot encode input
    input_encoded = pd.get_dummies(input_df)

    # Align with training columns
    for col in encoded_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[encoded_columns]

    # Scale input
    input_scaled = scaler.transform(input_encoded)

    # Predict
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)[0]

    result = label_encoder.inverse_transform(prediction)[0]
    prob = prediction_proba[prediction[0]] * 100

    # Display result
    st.markdown("---")
    st.markdown(f"<div class='result'>💰 <b>Predicted Income:</b> {result}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result'>📊 <b>Confidence:</b> {prob:.2f}%</div>", unsafe_allow_html=True)
