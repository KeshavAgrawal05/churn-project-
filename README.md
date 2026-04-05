# 📡 Customer Churn Prediction

> A production-ready machine learning system that predicts whether a telecom customer will churn, built with scikit-learn and deployed as an interactive Streamlit web app.

---

## 🗂️ Project Structure

```
churn_project/
│
├── data/
│   └── data.csv                  # Telco Customer Churn dataset (7,043 records)
│
├── notebooks/
│   └── churn_analysis.ipynb      # Exploratory Data Analysis + experiments
│
├── src/
│   ├── preprocessing.py          # Data cleaning, encoding, feature scaling
│   ├── train.py                  # Model training, evaluation, artifact saving
│   ├── predict.py                # Inference module for new customer records
│   └── utils.py                  # Shared helpers: I/O, metrics, plots
│
├── model/
│   ├── model.pkl                 # Trained best model (Random Forest)
│   ├── scaler.pkl                # Fitted StandardScaler
│   └── feature_names.pkl         # Ordered list of training feature names
│
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

