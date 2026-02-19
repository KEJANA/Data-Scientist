import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")
st.title("🚢 Titanic Survival Prediction")

MODEL_FILE = "aqi_random_forest_model.pkl"

# -----------------------------
# Load Model (Safe Fallback)
# -----------------------------
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    st.warning("Model file not found. Using default trained model.")
    
    # Dummy minimal training (fallback so app never crashes)
    data = {
        "Pclass": [1, 3, 2, 1],
        "Age": [22, 38, 26, 35],
        "SibSp": [1, 1, 0, 1],
        "Parch": [0, 0, 0, 0],
        "Fare": [7.25, 71.28, 7.92, 53.10],
        "Embarked": [2, 0, 2, 2],
        "Survived": [0, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    model = RandomForestClassifier()
    model.fit(X, y)

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

# Encoding same as training
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked_val = embarked_map[embarked]

# -----------------------------
# Input DataFrame
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
    result = model.predict(input_data)[0]

    if result == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("😢 Passenger Did Not Survive")
