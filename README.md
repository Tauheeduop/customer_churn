# 🧾 Customer Churn Prediction Project

---

## 📌 Project Overview

The **Customer Churn Prediction** project aims to predict whether a customer is likely to **leave (churn)** or **stay** with a company using **Machine Learning**.

It uses **classification algorithms** to analyze customer behavior and identify key patterns that contribute to churn.  
This project is a great starting point for understanding the **end-to-end ML workflow**, including **data preprocessing, model training, evaluation, and deployment readiness**.

---

## 🧠 Objectives

- Understand and apply key **data preprocessing** techniques  
- Train multiple **classification models** to predict churn  
- Evaluate models using **precision, recall, F1-score, and accuracy**  
- Prepare the project for integration into an **Agentic AI System (v2)**  

---

## 🧰 Tech Stack

- **Python 3.10+**
- **Libraries:**  
  `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- *(Optional next step)*: **FastAPI** & **Streamlit** for deployment  
- **Jupyter Notebook / VS Code** for development  

---

## 📂 Project Structure
customer_churn/
│
├── data/
│ └── churn.csv # Dataset
│
├── main.py # Main training & evaluation script
│
├── model/ # (Optional) saved model directory
│
├── requirements.txt # Dependencies list
│
└── README.md # Project documentation


---

## 🔍 Workflow

### 🧹 Data Preprocessing
- Handle missing or invalid values  
- Encode categorical variables (`LabelEncoder`)  
- Apply feature scaling (`StandardScaler`)  

### 🧩 Train-Test Split
- Split dataset into **80% training** and **20% testing** using `train_test_split`.

### 🧮 Model Training
Train multiple classification models:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- SVM  

### 📊 Model Evaluation
Compare models using:
- **Accuracy**
- **Confusion Matrix**
- **Classification Report**

### 🤖 Next Step (v2 — Agentic System)
- Integrate with **FastAPI** + **Streamlit**  
- Build an **autonomous churn prediction agent**  
- Automate retention actions (emails, alerts, CRM updates)

---

## 📈 Sample Output
