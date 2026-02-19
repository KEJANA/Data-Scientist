import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("aqi_random_forest_model.pkl")

st.title("🚢 Titanic Survival Prediction (Random Forest)")
st.write("Enter passenger details to predict survival")

# Input fields (numeric features only)
Pclass = st.number_input("Passenger Class (1-3)", min_value=1, max_value=3, value=1)
Age = st.number_input("Age", min_value=0.0, value=25.0)
SibSp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0)
Parch = st.number_input("Parents/Children Aboard", min_value=0, value=0)
Fare = st.number_input("Fare", min_value=0.0, value=50.0)

# Predict button
if st.button("Predict Survival"):
    input_data = np.array([[Pclass, Age, SibSp, Parch, Fare]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")