import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("aqi_random_forest_model.pkl")

st.title("🚢 Titanic Survival Prediction (Random Forest)")

st.write("Enter passenger details to predict survival:")

# -----------------------------
# User Inputs
# -----------------------------
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
sibsp = st.slider("Siblings / Spouses Aboard", 0, 5, 0)
parch = st.slider("Parents / Children Aboard", 0, 5, 0)
fare = st.slider("Fare", 0, 500, 50)
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# Convert embarked to numeric (same training assumption)
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_map[embarked]

# -----------------------------
# Prepare Input Data
# (Name and Sex were dropped in notebook)
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
if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("😢 Passenger Did Not Survive")
