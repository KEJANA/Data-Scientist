import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("🚢 Titanic Survival Prediction App")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

# -----------------------------
# Preprocessing (same as notebook)
# -----------------------------
df = df.drop(["Name", "Sex"], axis=1)

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Convert Embarked to numeric
df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train Model
model = RandomForestClassifier()
model.fit(X, y)

# -----------------------------
# User Input
# -----------------------------
st.sidebar.header("Enter Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
age = st.sidebar.slider("Age", 0, 80, 25)
sibsp = st.sidebar.slider("Siblings / Spouses", 0, 5, 0)
parch = st.sidebar.slider("Parents / Children", 0, 5, 0)
fare = st.sidebar.slider("Fare", 0, 500, 50)
embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"])

embarked = {"C": 0, "Q": 1, "S": 2}[embarked]

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
        st.success("🎉 Passenger Survived!")
    else:
        st.error("😢 Passenger Did Not Survive")
