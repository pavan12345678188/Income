import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import pickle

# Load trained components
with open("boostxg_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("xgscaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("xgbolabel_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

with open("model_features.pkl", "rb") as features_file:
    model_features = pickle.load(features_file)

# Load original data (for input options)
df = pd.read_csv("income_evaluation.csv")
df.columns = df.columns.str.strip()

# Page styling
st.set_page_config(page_title="Income Predictor", layout="centered")

st.markdown("""
    <style>
        .main { background-color: #f0f2f6; padding: 2rem; border-radius: 10px; }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 8px;
        }
        .result {
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💼 Income Prediction App")
st.subheader("Will this person earn more than 50K?")

# Input form
def get_user_input():
    with st.form("user_form"):
        age = st.slider("Age", 17, 90, 30)
        workclass = st.selectbox("Workclass", df["workclass"].dropna().unique())
        education = st.selectbox("Education", df["education"].dropna().unique())
        marital_status = st.selectbox("Marital Status", df["marital-status"].dropna().unique())
        occupation = st.selectbox("Occupation", df["occupation"].dropna().unique())
        relationship = st.selectbox("Relationship", df["relationship"].dropna().unique())
        race = st.selectbox("Race", df["race"].dropna().unique())
        sex = st.selectbox("Sex", df["sex"].dropna().unique())
        hours_per_week = st.slider("Hours per Week", 1, 99, 40)
        native_country = st.selectbox("Native Country", df["native-country"].dropna().unique())

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

# Handle form
input_df, submitted = get_user_input()

if submitted:
    # Combine input with original dataset to align dummy variables
    combined_df = pd.concat([input_df, df.drop("income", axis=1)], axis=0)
    combined_encoded = pd.get_dummies(combined_df)

    # Add missing columns
    for col in model_features:
        if col not in combined_encoded.columns:
            combined_encoded[col] = 0

    # Ensure column order matches training
    final_input = combined_encoded[model_features].iloc[[0]]

    # Scale input
    scaled_input = scaler.transform(final_input)

    # Predict
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][prediction[0]] * 100
    result = label_encoder.inverse_transform(prediction)[0]

    # Show results
    st.markdown("---")
    st.markdown(f"<div class='result'>💰 <b>Predicted Income:</b> {result}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result'>📊 <b>Confidence:</b> {probability:.2f}%</div>", unsafe_allow_html=True)
