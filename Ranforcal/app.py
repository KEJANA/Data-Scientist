import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")
st.title("🚢 Titanic Survival Prediction (Random Forest)")

# -----------------------------
# Upload Model File
# -----------------------------
uploaded_model = st.file_uploader(
    "Upload your trained model file (.pkl)", type=["pkl"]
)

if uploaded_model is not None:
    model = joblib.load(uploaded_model)
    st.success("Model loaded successfully!")
else:
    st.warning("Please upload 'aqi_random_forest_model.pkl' to use your trained model.")
    st.stop()

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

# Encoding (same as training)
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked_val = embarked_map[embarked]

# -----------------------------
# Prepare Input Data
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
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("😢 Passenger Did Not Survive")
