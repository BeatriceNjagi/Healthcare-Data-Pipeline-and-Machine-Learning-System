from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.train import train

# Define the DAG
with DAG(
    dag_id   = "retrain_healthcare_model",
    schedule = "0 12 * * 6",        # Every Saturday at 12:00 noon
    start_date = datetime(2024, 1, 1),
    catchup  = False,
    tags     = ["healthcare", "ml"]
) as dag:

    retrain_task = PythonOperator(
        task_id      = "retrain_model",
        python_callable = train
    )