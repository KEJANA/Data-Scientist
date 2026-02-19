import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")
st.title("🚢 Titanic Survival Prediction (Random Forest)")

# -----------------------------
# Find Model Path (robust for Streamlit Cloud)
# -----------------------------
possible_paths = [
    "aqi_random_forest_model.pkl",
    "models/aqi_random_forest_model.pkl",
    "./aqi_random_forest_model.pkl",
    "./models/aqi_random_forest_model.pkl"
]

model_path = None
for path in possible_paths:
    if os.path.exists(path):
        model_path = path
        break

if model_path is None:
    st.error("❌ Model file 'aqi_random_forest_model.pkl' not found. Upload it to repo root or models/ folder.")
    st.stop()

# Load model
model = joblib.load(model_path)

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
embarked_val = embarked_map[embarked]

# -----------------------------
# Prepare Input Data
# (Name and Sex were dropped during training)
# -----------------------------
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked_val]
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
