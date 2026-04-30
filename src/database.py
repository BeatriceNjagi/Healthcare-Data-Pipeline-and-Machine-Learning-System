import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_engine():
    """
    Creates and returns a SQLAlchemy engine using the DATABASE_URL
    environment variable. On Render this is set in the dashboard.
    On local it is read from the .env file.
    """
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set.")
    return create_engine(DATABASE_URL)


def load_raw_data(csv_path="data/healthcare_dataset.csv"):
    """
    Loads the raw CSV into the database as the raw_patients table.
    Uses if_exists='replace' so it is safe to run multiple times.
    """
    engine = get_engine()
    df = pd.read_csv(csv_path)

    # Standardise column names to lowercase with underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df.to_sql("raw_patients", engine, if_exists="replace", index=False)
    print(f"Loaded {len(df)} raw records into raw_patients table.")


def load_cleaned_data(df):
    """
    Loads a cleaned dataframe into the database as the cleaned_patients table.
    Called from clean.py after cleaning is complete.
    """
    engine = get_engine()
    df.to_sql("cleaned_patients", engine, if_exists="replace", index=False)
    print(f"Loaded {len(df)} cleaned records into cleaned_patients table.")


def fetch_data():
    """
    Fetches all records from the cleaned_patients table.
    Called from train.py to get the latest data for training.
    Returns a pandas dataframe.
    """
    engine = get_engine()

    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM cleaned_patients", conn)

    print(f"Fetched {len(df)} records from cleaned_patients table.")
    return df