# 📡 End-to-End ML Pipeline for Customer Churn Prediction

## 📋 Task Overview
This project implements a **production-ready machine learning pipeline** to predict customer churn for a telecommunications company. By utilizing scikit-learn's **Pipeline API**, the project ensures that preprocessing and model logic are bundled together, preventing data leakage and ensuring easy deployment.

## 🎯 Objective
As part of the AI/ML Engineering Internship at **DevelopersHub Corporation**, this task demonstrates:
* **Automated Preprocessing:** Handling numerical and categorical data seamlessly.
* **Model Benchmarking:** Comparing Logistic Regression and Random Forest.
* **Hyperparameter Optimization:** Using GridSearchCV for peak performance.
* **Serialization:** Exporting the entire end-to-end pipeline for production use.

## 📊 Dataset Specifications
* **Source:** Telco Customer Churn Dataset (`churn-bigml-80.csv`)
* **Size:** 2,666 customer records
* **Target Variable:** `Churn` (Boolean: True/False)
* **Key Features:** * **Account Info:** State, Area code, Account length.
    * **Plans:** International plan, Voice mail plan.
    * **Usage:** Total day/eve/night/intl (minutes, calls, and charges).
    * **Engagement:** Customer service calls.

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Machine Learning:** `scikit-learn` (Pipeline, ColumnTransformer, GridSearchCV)
* **Data Handling:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Deployment Tools:** `joblib` for model serialization

---

## 🔄 Pipeline Architecture



### 1. Preprocessing Layer
The data is split into two specialized streams before reaching the model:
* **Numerical Pipeline:** Imputation (median) → Standard Scaling.
* **Categorical Pipeline:** Imputation (constant) → One-Hot Encoding (handling unknown categories).

### 2. Model Selection & Tuning
We implemented a grid search across multiple algorithms:
* **Logistic Regression:** Optimized for regularization strength.
* **Random Forest:** Optimized for tree depth and estimator count.
* **Metric:** Optimized for **ROC-AUC** to balance precision and recall.

## 📈 Model Performance Summary
| Model | Accuracy | ROC-AUC |
| :--- | :--- | :--- |
| **Logistic Regression** | ~86% | ~87% |
| **Random Forest** | **~94%** | **~92%** |

> **Insight:** Random Forest significantly outperformed Logistic Regression, particularly in identifying complex non-linear patterns in usage charges and service call frequencies.

---

## 🚀 Installation & Usage

### 1. Clone & Install
```bash
git clone <your-repo-link>
cd End-to-End-ML-Pipeline
pip install -r requirements.txt