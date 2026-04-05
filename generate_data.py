"""
Script to generate a realistic synthetic Telco Customer Churn dataset.
Run this once to produce data/data.csv before training.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 7043

def generate_churn_dataset(n=N):
    gender = np.random.choice(["Male", "Female"], n)
    senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], n)
    dependents = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
    tenure = np.random.randint(1, 73, n)
    phone_service = np.random.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multiple_lines = np.where(
        phone_service == "No", "No phone service",
        np.random.choice(["Yes", "No"], n)
    )
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    online_security = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n)
    )
    online_backup = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n)
    )
    device_protection = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n)
    )
    tech_support = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n)
    )
    streaming_tv = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n)
    )
    streaming_movies = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], n)
    )
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )

    # Monthly charges correlated with services
    base_charge = np.random.uniform(18, 30, n)
    fiber_addon = (internet_service == "Fiber optic") * np.random.uniform(20, 40, n)
    dsl_addon = (internet_service == "DSL") * np.random.uniform(10, 20, n)
    monthly_charges = np.round(base_charge + fiber_addon + dsl_addon + np.random.uniform(0, 10, n), 2)

    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 5, n), 2)
    total_charges = np.clip(total_charges, 0, None)

    # Churn probability – higher for fiber, month-to-month, high charges, low tenure
    churn_prob = (
        0.05
        + (internet_service == "Fiber optic") * 0.15
        + (contract == "Month-to-month") * 0.20
        + (senior == 1) * 0.05
        + (tenure < 12) * 0.10
        + (monthly_charges > 70) * 0.10
        - (contract == "Two year") * 0.10
        - (tenure > 36) * 0.08
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.85)
    churn = np.where(np.random.rand(n) < churn_prob, "Yes", "No")

    customer_ids = [f"CUST-{str(i).zfill(5)}" for i in range(1, n + 1)]

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn,
    })

    # Inject ~1% missing values in TotalCharges (mimicking real Telco dataset)
    missing_idx = np.random.choice(df.index, size=int(n * 0.01), replace=False)
    df.loc[missing_idx, "TotalCharges"] = np.nan

    return df


if __name__ == "__main__":
    df = generate_churn_dataset()
    df.to_csv("data.csv", index=False)
    print(f"Dataset saved: {len(df)} rows, {df['Churn'].value_counts().to_dict()}")
