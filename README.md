```markdown
# End-to-End ML Pipeline for Customer Churn Prediction

## 📋 Task Overview
This project implements a production-ready machine learning pipeline for predicting customer churn in a telecommunications company. The pipeline is built using scikit-learn's Pipeline API, ensuring reusability and maintainability.

## 🎯 Objective
Build a reusable and production-ready machine learning pipeline that can:
- Preprocess customer data automatically
- Train multiple classification models
- Perform hyperparameter tuning
- Export the complete pipeline for deployment

## 📊 Dataset
**Dataset:** Telco Customer Churn Dataset (churn-bigml-80.csv)
- **Records:** 2,666 customers
- **Features:** 20 attributes including:
  - Customer demographics (State, Area code)
  - Account information (Account length, International plan, Voice mail plan)
  - Usage patterns (Total day/minutes/calls/charge, Evening/Night/International usage)
  - Customer service interactions (Customer service calls)
  - **Target:** Churn (Boolean: True/False)

## 🛠️ Technologies Used
- **Python 3.12**
- **scikit-learn** - ML pipeline, preprocessing, models, GridSearchCV
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical operations
- **matplotlib/seaborn** - Data visualization
- **joblib** - Model export and serialization

## 📁 Project Structure
```
├── churn_prediction_pipeline.ipynb  # Main Jupyter notebook
├── best_churn_prediction_pipeline_*.joblib  # Exported pipeline (generated)
├── README.md  # Project documentation
└── requirements.txt  # Project dependencies
```

## 🔄 Pipeline Components

### 1. Data Preprocessing
- **Numerical Features** (12 features):
  - Account length, Number vmail messages
  - Total day/eve/night/intl minutes
  - Total day/eve/night/intl calls
  - Total day/eve/night/intl charge
  - Customer service calls
  
- **Categorical Features** (5 features):
  - State
  - Area code (treated as categorical)
  - International plan
  - Voice mail plan
  - Boolean features automatically encoded

### 2. Preprocessing Steps
```python
Numerical Pipeline:
- SimpleImputer(strategy='median')
- StandardScaler()

Categorical Pipeline:
- SimpleImputer(strategy='constant', fill_value='missing')
- OneHotEncoder(handle_unknown='ignore')
```

### 3. Models Trained
- **Logistic Regression** with hyperparameter tuning
- **Random Forest Classifier** with hyperparameter tuning

### 4. Hyperparameter Tuning
- **GridSearchCV** with 5-fold cross-validation
- Scoring metric: ROC-AUC
- Parameter grids for both models

## 📈 Model Performance

### Logistic Regression
- **Accuracy:** ~85-87%
- **ROC-AUC:** ~85-88%

### Random Forest
- **Accuracy:** ~92-95%
- **ROC-AUC:** ~90-94%

*Note: Actual metrics may vary based on random seed and data split*

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Steps
1. **Clone the repository**
2. **Place the dataset** in `D:/download/churn-bigml-80.csv` (or update path in notebook)
3. **Run the Jupyter notebook** cells in order:
   - Cells 1-5: Data loading and preprocessing
   - Cells 6-7: Pipeline creation
   - Cells 8-10: GridSearchCV hyperparameter tuning
   - Cells 11-13: Model evaluation and comparison
   - Cells 14-15: Model export and testing
   - Cells 16-17: Feature importance and summary

### Quick Start
```python
# Load the saved pipeline
import joblib
model = joblib.load('best_churn_prediction_pipeline_Random_Forest.joblib')

# Make predictions on new data
predictions = model.predict(new_customer_data)
probabilities = model.predict_proba(new_customer_data)
```

## 💾 Model Export
The complete pipeline is exported using `joblib`:
```python
joblib.dump(best_model, 'best_churn_prediction_pipeline_{model_name}.joblib')
```

The exported file includes:
- All preprocessing steps
- Trained model with best parameters
- Feature names and encodings

## 📊 Key Insights

1. **Churn Rate:** ~14.5% of customers churn
2. **Important Features:**
   - Customer service calls count
   - Total day minutes/charge
   - International plan subscription
   - Voice mail plan subscription
3. **Risk Factors:**
   - High number of customer service calls
   - International plan subscribers
   - High total day usage

## 🔧 Future Improvements

1. **Model Enhancements:**
   - Try XGBoost or LightGBM
   - Implement ensemble methods
   - Handle class imbalance with SMOTE

2. **Feature Engineering:**
   - Create interaction features
   - Aggregate usage patterns
   - Time-based features

3. **Deployment:**
   - Create REST API using Flask/FastAPI
   - Build web interface with Streamlit/Gradio
   - Deploy to cloud platforms (AWS/GCP/Azure)

4. **Monitoring:**
   - Track model performance over time
   - Implement data drift detection
   - Set up automated retraining

## 📝 Requirements
Create a `requirements.txt` file:
```
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
joblib==1.3.1
jupyter==1.0.0
```

## 📚 Skills Gained
- ✅ ML pipeline construction with scikit-learn
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Model export and reusability
- ✅ Production-readiness practices
- ✅ Data preprocessing and feature engineering
- ✅ Model evaluation and comparison
- ✅ Visualization of results and insights

## 👨‍💻 Author
Developed as part of AI/ML Engineering Internship at DevelopersHub Corporation.

## 📄 License
This project is for educational purposes as part of the internship program.

---
**Note:** This pipeline is production-ready and can be easily integrated into any web application or API service for real-time customer churn predictions.
```