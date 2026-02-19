import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

st.title("🚢 Titanic Survival Prediction (Random Forest)")

# -----------------------------
# Load Model Safely
# -----------------------------
MODEL_PATH = "aqi_random_forest_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file 'aqi_random_forest_model.pkl' not found in repository.")
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# User Inputs
# -----------------------------
st.sidebar.header("Enter Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
age = st.sidebar.slider("Age", 0, 80, 25)
sibsp = st.sidebar.slider("Siblings / Spouses", 0, 5, 0)
parch = st.sidebar.slider("Parents / Children", 0, 5, 0)
fare = st.sidebar.slider("Fare", 0, 500, 50)
embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"])

# same encoding used during training
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_map[embarked]

# -----------------------------
# Prepare Input Data
# (Name and Sex were dropped in training)
# -----------------------------
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked]
})

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Survival"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("😢 Passenger Did Not Survive")
