import streamlit as st
import joblib
import numpy as np
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
    }
    .sub-text {
        text-align: center;
        font-size: 18px;
        color: gray;
    }
    .card {
        background-color: #1e1e2f;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Models
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
linear_path = os.path.join(BASE_DIR, "linear_model.pkl")
rf_path = os.path.join(BASE_DIR, "Randomforest.pkl")

@st.cache_resource
def load_models():
    linear_model = joblib.load(linear_path)
    rf_model = joblib.load(rf_path)
    return linear_model, rf_model

linear_model, rf_model = load_models()

# -----------------------------
# Title Section
# -----------------------------
st.markdown('<div class="main-title">🏠 House Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Predict house price using ML models</div>', unsafe_allow_html=True)
st.write("")

# -----------------------------
# Card Layout
# -----------------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Model Selection
    model_choice = st.selectbox(
        "🤖 Choose Prediction Model",
        ["Linear Regression", "Random Forest"]
    )

    # Input Columns
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "📐 Area (sq.ft)",
            min_value=100,
            max_value=10000,
            value=1000
        )

    with col2:
        bedrooms = st.number_input(
            "🛏️ Bedrooms",
            min_value=1,
            max_value=10,
            value=2
        )

    st.write("")

    # Predict Button
    predict_btn = st.button("🚀 Predict Price")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction Result
# -----------------------------
if predict_btn:
    input_data = np.array([[area, bedrooms]])

    if model_choice == "Linear Regression":
        prediction = linear_model.predict(input_data)[0]
        model_name = "Linear Regression"
    else:
        prediction = rf_model.predict(input_data)[0]
        model_name = "Random Forest"

    st.markdown("### 📊 Prediction Result")
    st.success(f"🏷️ Model Used: {model_name}")
    st.metric("💰 Estimated Price (Lakhs)", f"₹ {prediction:.2f}")