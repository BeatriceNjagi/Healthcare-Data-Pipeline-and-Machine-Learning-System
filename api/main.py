import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import gc

app = FastAPI(
    title       = "Healthcare Analytics API",
    description = "Predicts patient test results: Normal, Abnormal, or Inconclusive",
    version     = "1.0.0"
)

MODELS_DIR  = "models"
NUMERIC_COLS = ["age", "billing_amount"]

# Global variables for lazy loading
# Models are loaded once on first request and reused
model           = None
encoders        = None
scaler          = None
feature_columns = None


def get_artifacts():
    """
    Loads all model artifacts from the models/ folder.
    Uses lazy loading — only loads once, reuses on subsequent requests.
    Raises a clear error if models folder or files are missing.
    """
    global model, encoders, scaler, feature_columns

    if model is None:
        if not os.path.exists(MODELS_DIR):
            raise HTTPException(
                status_code = 503,
                detail      = "Models folder not found. Run src/train.py first."
            )

        required_files = [
            "best_model.joblib",
            "encoders.joblib",
            "scaler.joblib",
            "feature_columns.joblib"
        ]

        missing = [f for f in required_files
                   if not os.path.exists(os.path.join(MODELS_DIR, f))]

        if missing:
            raise HTTPException(
                status_code = 503,
                detail      = f"Missing model files: {missing}. Run src/train.py first."
            )

        model           = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
        encoders        = joblib.load(os.path.join(MODELS_DIR, "encoders.joblib"))
        scaler          = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))

    return model, encoders, scaler, feature_columns


class PatientInput(BaseModel):
    Age               : int
    Gender            : str
    Blood_Type        : str
    Medical_Condition : str
    Billing_Amount    : float
    Admission_Type    : str
    Insurance_Provider: str
    Medication        : str


@app.get("/")
def root():
    return {"message": "Healthcare Analytics API is live 🏥"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug")
def debug():
    """
    Returns the state of the models folder.
    Use this endpoint to diagnose deployment issues.
    """
    files = os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else []
    return {
        "models_folder_exists" : os.path.exists(MODELS_DIR),
        "files_in_models"      : files,
        "model_loaded"         : model is not None
    }


@app.post("/predict")
def predict(patient: PatientInput):
    """
    Predicts the test result for a patient.

    Input: patient demographics and clinical info
    Output: predicted_test_result — Normal, Abnormal, or Inconclusive
    """
    try:
        model, encoders, scaler, feature_columns = get_artifacts()

        categorical_map = {
            "gender"            : patient.Gender,
            "blood_type"        : patient.Blood_Type,
            "medical_condition" : patient.Medical_Condition,
            "admission_type"    : patient.Admission_Type,
            "insurance_provider": patient.Insurance_Provider,
            "medication"        : patient.Medication,
        }

        # Step 1 — encode categorical values
        encoded = {}
        for col, val in categorical_map.items():
            le        = encoders[col]
            val_clean = val.strip().title()

            if val_clean not in le.classes_:
                raise HTTPException(
                    status_code = 422,
                    detail      = f"Unknown value '{val}' for '{col}'. "
                                  f"Valid options: {list(le.classes_)}"
                )

            encoded[col] = le.transform([val_clean])[0]

        # Step 2 — build features dataframe
        features = pd.DataFrame([{
            "age"               : patient.Age,
            "gender"            : encoded["gender"],
            "blood_type"        : encoded["blood_type"],
            "medical_condition" : encoded["medical_condition"],
            "insurance_provider": encoded["insurance_provider"],
            "billing_amount"    : patient.Billing_Amount,
            "admission_type"    : encoded["admission_type"],
            "medication"        : encoded["medication"],
        }])

        # Step 3 — scale numeric columns
        features[NUMERIC_COLS] = scaler.transform(features[NUMERIC_COLS])

        # Step 4 — reindex to match exact training column order
        features = features.reindex(columns=feature_columns)

        # Step 5 — predict
        prediction = model.predict(features)[0]

        # Step 6 — decode prediction back to label
        label = encoders["target"].inverse_transform([prediction])[0]

        gc.collect()

        return {"predicted_test_result": label}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)