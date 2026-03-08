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
* **Key Features:**
    * **Account Info:** State, Area code, Account length.
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

```

### 2. Run the Notebook

Open `churn_prediction_pipeline.ipynb` in Jupyter or VS Code. Ensure your dataset is placed in the directory specified in the data loading cell.

### 3. Production Inference

You can load the saved pipeline in any Python script without needing to manually preprocess new data:

```python
import joblib

# Load the complete production-ready pipeline
model = joblib.load('best_churn_prediction_pipeline_Random_Forest.joblib')

# Predict directly on raw data
predictions = model.predict(new_customer_df)

```

## 📂 Project Structure

```text
├── churn_prediction_pipeline.ipynb   # Main development notebook
├── best_churn_prediction_pipeline.joblib # Final serialized model
├── README.md                         # Project documentation
└── requirements.txt                  # Dependency list

```

---

## 📊 Key Insights

* **Churn Rate:** Approximately **14.5%** of the customer base in this dataset churns, indicating a class imbalance that the model must navigate.
* **Important Features:** The model identified the following as the strongest predictors of churn:
* **Customer Service Calls:** The frequency of interactions with support.
* **Total Day Usage:** Both total minutes and the resulting charges.
* **Plan Subscriptions:** Whether the user has an International or Voice Mail plan.


* **Primary Risk Factors:** Customers are significantly more likely to leave if they exhibit:
* A high volume of customer service calls (potential dissatisfaction).
* Subscription to an International plan.
* Abnormally high daytime usage.



---

## 🔧 Future Improvements

### 1. Model & Data Enhancements

* **Advanced Algorithms:** Experiment with Gradient Boosting machines like **XGBoost**, **LightGBM**, or **CatBoost**.
* **Imbalance Handling:** Implement **SMOTE** (Synthetic Minority Over-sampling Technique) to better represent the minority churn class.
* **Feature Engineering:** Create interaction features (e.g., charge per minute) and aggregate usage patterns over time.

### 2. Deployment & MLOps

* **API Development:** Wrap the `joblib` pipeline in a **FastAPI** or **Flask** REST API for real-time predictions.
* **User Interface:** Build a dashboard using **Streamlit** or **Gradio** to allow non-technical stakeholders to input customer data.
* **Cloud Integration:** Deploy the model to **AWS (EC2/Sagemaker)** or **Azure** to handle production workloads.
* **Monitoring:** Establish automated retraining loops and data drift detection to ensure the model remains accurate as customer behavior evolves.

---

## 👨‍💻 Author

**Developed as part of the AI/ML Engineering Internship at DevelopersHub Corporation.**

## 📄 License

This project is for educational purposes as part of the internship program.

```

Would you like me to generate the `requirements.txt` file content as well to complete your repository?

```