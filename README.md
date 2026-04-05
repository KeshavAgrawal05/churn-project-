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

## 🚀 Quick Start

### 1. Clone / download the project

```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python src/train.py
```

This will:
- Load and preprocess `data/data.csv`
- Train Logistic Regression and Random Forest
- Print accuracy + classification report for both
- Save the best model to `model/`

### 5. Run the web app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🧠 How It Works

### Pipeline

```
Raw CSV  →  Clean  →  Encode  →  Scale  →  Train  →  Predict
```

| Step | Description |
|------|-------------|
| **Load** | Read CSV, inspect shape and dtypes |
| **Clean** | Fix `TotalCharges` blanks → median imputation |
| **Binary encode** | `Yes/No` → `1/0` for 5 columns |
| **One-Hot encode** | 11 categorical columns (drop_first=True) |
| **Scale** | StandardScaler on `tenure`, `MonthlyCharges`, `TotalCharges` |
| **Train** | Logistic Regression + Random Forest (class_weight="balanced") |
| **Evaluate** | Accuracy, Confusion Matrix, Classification Report |
| **Save** | model.pkl, scaler.pkl, feature_names.pkl |

### Model Comparison

| Model | Test Accuracy |
|-------|--------------|
| Logistic Regression | ~67% |
| **Random Forest** ✓ | **~78%** |

Random Forest is selected automatically as the best model.

---

## 📊 Key Features

- **Churn distribution visualisation** — understand class imbalance upfront
- **Feature importance plot** — top 15 drivers of churn from Random Forest
- **Model comparison chart** — side-by-side accuracy bar chart
- **Confusion matrix** — TP / FP / TN / FN breakdown
- **Interactive Streamlit UI** — input any customer profile and get instant predictions

---

## 🗒️ Standalone Prediction

```python
from src.predict import predict_churn

customer = {
    "gender":           "Male",
    "SeniorCitizen":    0,
    "Partner":          "Yes",
    "Dependents":       "No",
    "tenure":           12,
    "PhoneService":     "Yes",
    "MultipleLines":    "No",
    "InternetService":  "Fiber optic",
    "OnlineSecurity":   "No",
    "OnlineBackup":     "No",
    "DeviceProtection": "No",
    "TechSupport":      "No",
    "StreamingTV":      "Yes",
    "StreamingMovies":  "Yes",
    "Contract":         "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod":    "Electronic check",
    "MonthlyCharges":   85.00,
    "TotalCharges":     1020.00,
}

result = predict_churn(customer)
print(result)
# {'prediction': 1, 'label': 'Churn', 'probability': 0.73}
```

---

## 📸 Screenshots

```
[Screenshot 1] — Streamlit app landing page with dark UI
[Screenshot 2] — Customer form filled out
[Screenshot 3] — Prediction result: "Likely to Churn" with risk gauge
[Screenshot 4] — Feature importance bar chart
```

---

## 📌 Resume Bullet Points

Copy-paste these into your CV / LinkedIn:

- **Built end-to-end customer churn prediction system** using Random Forest and Logistic Regression on 7,000+ telecom records; achieved ~78% accuracy with balanced class handling
- **Engineered modular ML pipeline** (preprocessing, training, prediction modules) following production best practices: StandardScaler, One-Hot Encoding, pickle-based model persistence
- **Deployed interactive prediction dashboard** using Streamlit, enabling real-time churn risk assessment from customer inputs with probability scoring and risk-level indicators
- **Applied feature engineering** techniques including binary encoding, OHE, and numeric scaling; identified top churn drivers via Random Forest feature importance analysis

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0 | Data loading and manipulation |
| numpy | ≥1.24 | Numerical operations |
| scikit-learn | ≥1.3 | ML models, preprocessing, metrics |
| matplotlib | ≥3.7 | Visualisation |
| seaborn | ≥0.12 | Statistical plots |
| streamlit | ≥1.28 | Web application |

---

## 📄 License

MIT — free to use for personal or commercial projects.
