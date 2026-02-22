# 📍 K-Nearest Neighbors (KNN) Classification Model – Prediction App

> A Machine Learning web application that performs classification using the K-Nearest Neighbors (KNN) algorithm to predict categorical outcomes based on input features.

---

## 📌 Project Overview
This project implements a K-Nearest Neighbors (KNN) classification model to predict categorical labels based on similarity between data points.  
The trained model is deployed as an interactive Streamlit web application that enables real-time predictions through a simple user interface.

KNN is a distance-based algorithm that classifies a new data point by analyzing the nearest neighbors in the feature space.

---

## 🎯 Objectives
- Build a distance-based classification model using KNN  
- Predict categorical outcomes from input feature values  
- Deploy the model as an interactive Streamlit web app  
- Provide accurate real-time classification predictions  

---

## 🧠 Machine Learning Algorithm
### K-Nearest Neighbors (KNN)

KNN is a supervised learning algorithm that:
- Calculates distance between data points  
- Finds the ‘K’ nearest neighbors  
- Assigns the most frequent class among neighbors  

### Why KNN?
- Simple and intuitive algorithm  
- No complex training phase  
- Works well for pattern recognition tasks  
- Effective for small to medium-sized datasets  

---

## 📊 Dataset Description
The dataset contains multiple numerical and categorical features used to classify output categories such as:
- Disease Classification
- Customer Category Prediction
- Risk Level Classification
- Binary / Multi-class Classification Problems

The model predicts class labels based on similarity with nearest data points.

---

## 🛠️ Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  

---

## ⚙️ Model Development Workflow
1. Data Collection & Cleaning  
2. Feature Scaling (important for KNN)  
3. Model Training using KNeighborsClassifier  
4. Model Evaluation using Accuracy Metrics  
5. Deployment via Streamlit Web Application  

---

## 📈 Model Evaluation Metrics
- Accuracy Score  
- Precision  
- Recall  
- Confusion Matrix  

These metrics ensure reliable classification performance.

---

## 🚀 Application Features
- Real-time classification prediction  
- Interactive input interface  
- Distance-based similarity prediction  
- Lightweight and efficient model  

---
🔮 Future Enhancements
	•	Hyperparameter tuning for optimal K value
	•	Distance metric comparison (Euclidean, Manhattan)
	•	Feature importance visualization
	•	Cloud deployment for public access
  
👩‍💻 Author

Kejana V
B.Tech Artificial Intelligence & Data Science
Machine Learning & Data Science Enthusiast

📫 Email: kejav82@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/kejana-v-b2624533b/

## ▶️ How to Run the Application
```bash
pip install -r requirements.txt
streamlit run app.py
