import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import pickle

# Load model, scaler, and label encoder
with open("xgboost_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Load dataset to get input options
df = pd.read_csv("income_evaluation.csv")

# Custom Page Configuration
st.set_page_config(
    page_title="Income Prediction App",
    layout="centered",
    initial_sidebar_state="auto"
)

# Header with styling
st.markdown(
    """
    <style>
        .main {
            background-color: #f4f4f4;
            padding: 2rem;
            border-radius: 15px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
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
st.subheader("Predict whether income is >50K or <=50K based on demographics")

# Input form
def user_input_features():
    with st.form("income_form"):
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

        input_data = {
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

        return pd.DataFrame([input_data]) if submitted else None

input_df = user_input_features()

if input_df is not None:
    # Append to full dataframe for consistent encoding
    df_full = pd.concat([input_df, df], axis=0)
    df_full = pd.get_dummies(df_full)

    # Align with training columns
    input_processed = df_full[:1]

    # Scale input
    input_scaled = scaler.transform(input_processed)

    # Prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    predicted_class = label_encoder.inverse_transform(prediction)[0]
    prob = prediction_proba[0][prediction[0]] * 100

    # Display
    st.markdown("---")
    st.markdown(f"<div class='result'>💰 **Predicted Income:** {predicted_class}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result'>📊 **Confidence:** {prob:.2f}%</div>", unsafe_allow_html=True)
