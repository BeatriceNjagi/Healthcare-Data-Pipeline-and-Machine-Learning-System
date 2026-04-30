import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.database import load_raw_data, load_cleaned_data


def clean(csv_path="data/healthcare_dataset.csv"):
    """
    Full cleaning pipeline:
    1. Load raw CSV
    2. Load raw data into database
    3. Clean and transform
    4. Load cleaned data into database
    """

    print("Loading raw data...")
    df = pd.read_csv(csv_path)

    # Standardise column names — lowercase with underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    print(f"Raw shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # --- Load raw data into database first ---
    load_raw_data(csv_path)

    # --- Remove duplicates ---
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows.")

    # --- Handle missing values ---
    df = df.dropna()
    print(f"After dropping nulls: {df.shape}")

    # --- Drop irrelevant columns ---
    # Name and doctor are not useful for prediction
    cols_to_drop = ["name", "doctor", "hospital", "room_number",
                    "date_of_admission", "discharge_date"]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped columns: {cols_to_drop}")

    # --- Standardise categorical columns ---
    # Title case ensures consistency e.g. "male" and "Male" become "Male"
    categorical_cols = ["gender", "blood_type", "medical_condition",
                        "admission_type", "insurance_provider",
                        "medication", "test_results"]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # --- Validate billing amount ---
    # Negative billing amounts are invalid
    if "billing_amount" in df.columns:
        df = df[df["billing_amount"] > 0]

    # --- Validate age ---
    if "age" in df.columns:
        df = df[(df["age"] > 0) & (df["age"] < 120)]

    print(f"Final cleaned shape: {df.shape}")
    print(f"Test results distribution:\n{df['test_results'].value_counts()}")

    # --- Load cleaned data into database ---
    load_cleaned_data(df)

    print("Cleaning complete.")
    return df


if __name__ == "__main__":
    clean()

     
        
