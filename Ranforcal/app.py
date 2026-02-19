import streamlit as st
import joblib
import numpy as np
import os

st.title("🌫️ AQI Prediction - Random Forest")

# Correct model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "aqi_random_forest_model.pkl")

# Load model safely
model = joblib.load(MODEL_PATH)

st.write("Enter environmental parameters")

# Example inputs (change based on your dataset features)
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
feature4 = st.number_input("Feature 4")
feature5 = st.number_input("Feature 5")

if st.button("Predict AQI"):
    input_data = np.array([[feature1, feature2, feature3, feature4, feature5]])
    prediction = model.predict(input_data)
    st.success(f"Predicted AQI Category: {prediction[0]}")
