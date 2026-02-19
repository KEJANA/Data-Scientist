import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("titanic.csv")
    return df

df = load_data()

st.title("🚢 Titanic Survival Prediction App")

st.write("Dataset Preview:")
st.dataframe(df.head())

# -----------------------------
# Data Preprocessing
# -----------------------------
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna('S', inplace=True)

# Convert categorical to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# -----------------------------
# Train Model
# -----------------------------
X = df.drop("Survived", axis=1)
y = df["Survived"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# -----------------------------
# User Inputs
# -----------------------------
st.sidebar.header("Enter Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 25)
sibsp = st.sidebar.slider("Siblings/Spouses", 0, 5, 0)
parch = st.sidebar.slider("Parents/Children", 0, 5, 0)
fare = st.sidebar.slider("Fare", 0, 500, 50)
embarked = st.sidebar.selectbox("Embarked", ["S", "C", "Q"])

# Encode inputs
sex = 1 if sex == "female" else 0
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0

input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked_C': [embarked_C],
    'Embarked_Q': [embarked_Q]
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
