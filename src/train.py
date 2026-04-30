import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.database import fetch_data

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

CATEGORICAL_COLS = ["gender", "blood_type", "medical_condition",
                    "admission_type", "insurance_provider", "medication"]
NUMERIC_COLS     = ["age", "billing_amount"]
TARGET           = "test_results"


def preprocess(df):
    """
    Encodes categorical columns and scales numeric columns.
    Returns features X, target y, encoders dict, and scaler.
    """
    df = df.copy()

    # Drop any leftover identity or date columns
    df.drop(columns=["id", "inserted_at"], errors="ignore", inplace=True)

    encoders = {}

    # Encode each categorical column with a LabelEncoder
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Encode target column
    target_le = LabelEncoder()
    df[TARGET] = target_le.fit_transform(df[TARGET])
    encoders["target"] = target_le

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y, encoders


def evaluate(name, model, X_test, y_test, target_le):
    """
    Evaluates a model and prints all required metrics.
    Returns the weighted F1 score for model comparison.
    """
    y_pred  = model.predict(X_test)
    labels  = target_le.classes_

    print(f"\n{'='*40}")
    print(f" Model: {name}")
    print(f"{'='*40}")
    print(f" Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f" Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f" Recall   : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f" F1 Score : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return f1_score(y_test, y_pred, average="weighted", zero_division=0)


def train():
    """
    Full training pipeline:
    1. Fetch data from database
    2. Preprocess
    3. Train three models
    4. Select best model by F1 score
    5. Save all artifacts
    """

    print(f"\n[{datetime.now()}] Starting training...")

    # --- Fetch data from database ---
    df = fetch_data()
    print(f"Fetched {len(df)} records.")

    # --- Preprocess ---
    X, y, encoders = preprocess(df)

    # Save feature columns before splitting
    # This ensures prediction uses exact same column order as training
    feature_columns = X.columns.tolist()
    print(f"Features: {feature_columns}")

    # --- Train test split ---
    # Stratify ensures balanced classes in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Scale numeric columns ---
    # Fit scaler on train only — never on test data
    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS]  = scaler.transform(X_test[NUMERIC_COLS])

    target_le = encoders["target"]

    # --- Define models ---
    models = {
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost"             : XGBClassifier(eval_metric="mlogloss", random_state=42)
    }

    # --- Train and evaluate each model ---
    best_name, best_model, best_f1 = None, None, 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        f1 = evaluate(name, model, X_test, y_test, target_le)

        if f1 > best_f1:
            best_f1   = f1
            best_name = name
            best_model = model

    print(f"\n✅ Best Model: {best_name} (F1={best_f1:.4f})")

    # --- Save all artifacts ---
    joblib.dump(best_model,      f"{MODELS_DIR}/best_model.joblib",      compress=3)
    joblib.dump(encoders,        f"{MODELS_DIR}/encoders.joblib",        compress=3)
    joblib.dump(scaler,          f"{MODELS_DIR}/scaler.joblib",          compress=3)
    joblib.dump(feature_columns, f"{MODELS_DIR}/feature_columns.joblib", compress=3)

    print(f"All artifacts saved to {MODELS_DIR}/")


if __name__ == "__main__":
    train()
