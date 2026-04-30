# Healthcare Analytics System

An end-to-end ML system that predicts patient test results as **Normal**, **Abnormal**, or **Inconclusive** using a synthetic healthcare dataset.

---

## Project Structure

```
Healthcare-Data-Pipeline-and-Machine-Learning-System/
├── api/
│   └── main.py              
│
├── data/
│   └── healthcare_dataset.csv
│
├── models/
│   ├── best_model.joblib
│   ├── scaler.joblib
│   ├── encoders.joblib
│   ├── feature_columns.joblib
│   └── .gitkeep
│
├── src/
│   ├── clean.py             # Data cleaning & preprocessing
│   ├── database.py          # PostgreSQL connection & ingestion
│   ├── train.py             # Model training
│   └── retrain_dag.py       # Airflow DAG
│
├── requirements.txt         # Render dependencies
├── pyproject.toml           # uv/local dev dependencies
├── .env.example             # Env template
├── .gitignore
└── README.md
```
---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/BeatriceNjagi/Healthcare-Data-Pipeline-and-Machine-Learning-System
cd Healthcare-Data-Pipeline-and-Machine-Learning-System
```

### 2. Install Dependencies with uv
```bash
uv sync
```

### 3. Set Up Environment Variables
Create a file named .env
On Mac/Linux
```
touch .env
```
On Windows
```
New-Item .env -ItemType File
```
Open and edit the .env 
```
nano .env
```
Then add your database URL:
```
DATABASE_URL=postgresql://user:password@host:port/dbname
```
Save with Ctrl+O → Enter → Ctrl+X 

### 4. Load and clean raw data 
```bash
uv run python clean.py
```

### 5. Set Up Database and Load raw and cleaned Data
```bash
uv run python database.py
```

### 6. Train the Model
```bash
uv run python train.py
```

### 7. Run the API Locally
```bash
uvicorn main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

---

## API Usage

### POST /predict

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 45,
    "Gender": "Male",
    "Blood_Type": "O+",
    "Medical_Condition": "Diabetes",
    "Billing_Amount": 2000.5,
    "Admission_Type": "Emergency",
    "Insurance_Provider": "Cigna",
    "Medication": "Aspirin"
  }'
```

**Response:**
```json
{
  "predicted_test_result": "Abnormal"
}
```

### GET /health
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok"}
```

---

## Airflow DAG

The model retrains automatically every Saturday at 12:00 noon.

```bash
# Set airflow home
export AIRFLOW_HOME=$(pwd)/airflow

# Initialize database
uv run airflow db migrate

# Create admin user/password
uv run airflow standalone

# Start webserver
airflow webserver --port 808

# Start scheduler (new terminal)
airflow scheduler
```

Visit `http://localhost:8080` to see the DAG.
username:admin
password:in the simple_auth_manager_passwords file

---

## Models Used

| Model | Description |
|---|---|
| Logistic Regression | Simple baseline linear model |
| XGBoost | Gradient boosted trees (usually best) |

Evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.

---

## Live API

`https://healthcare-data-pipeline-and-machine.onrender.com`
