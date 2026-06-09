from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
from custom_transformers import winsorize, NamedWinsorizer


app = FastAPI(title="Loan Default Prediction API", version="1.0.0")




# -----------------------------------------------------------
# Load model AFTER defining NamedWinsorizer
# -----------------------------------------------------------
model = joblib.load("loan_pipeline.pkl")


# -----------------------------------------------------------
# Input schema
# -----------------------------------------------------------
class LoanFeatures(BaseModel):
    # Numerical
    loan_amount: int
    Credit_Score: int
    rate_of_interest: Optional[float] = None
    Interest_rate_spread: Optional[float] = None
    Upfront_charges: Optional[float] = None
    term: Optional[float] = None
    property_value: Optional[float] = None
    income: Optional[float] = None
    LTV: Optional[float] = None
    dtir1: Optional[float] = None

    # Categorical
    loan_limit: Optional[str] = None
    Gender: Optional[str] = None
    approv_in_adv: Optional[str] = None
    loan_type: Optional[str] = None
    loan_purpose: Optional[str] = None
    Credit_Worthiness: Optional[str] = None
    open_credit: Optional[str] = None
    business_or_commercial: Optional[str] = None
    Neg_ammortization: Optional[str] = None
    interest_only: Optional[str] = None
    lump_sum_payment: Optional[str] = None
    construction_type: Optional[str] = None
    occupancy_type: Optional[str] = None
    Secured_by: Optional[str] = None
    total_units: Optional[str] = None
    credit_type: Optional[str] = None
    co_applicant_credit_type: Optional[str] = None
    age: Optional[str] = None
    submission_of_application: Optional[str] = None
    Region: Optional[str] = None
    Security_Type: Optional[str] = None


def to_dataframe(features: LoanFeatures) -> pd.DataFrame:
    data = features.dict()
    data["co-applicant_credit_type"] = data.pop("co_applicant_credit_type")
    return pd.DataFrame([data])


# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Loan Default Prediction API is running.",
        "docs": "Visit /docs for Swagger UI"
    }


@app.post("/predict")
def predict(features: LoanFeatures):
    try:
        df = to_dataframe(features)
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0].tolist()

        return {
            "prediction": int(prediction),
            "label": "Default" if prediction == 1 else "No Default",
            "probability": {
                "no_default": round(probability[0], 4),
                "default": round(probability[1], 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(records: list[LoanFeatures]):
    try:
        df = pd.concat([to_dataframe(r) for r in records], ignore_index=True)
        predictions = model.predict(df).tolist()
        probabilities = model.predict_proba(df).tolist()

        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                "prediction": int(pred),
                "label": "Default" if pred == 1 else "No Default",
                "probability": {
                    "no_default": round(prob[0], 4),
                    "default": round(prob[1], 4)
                }
            })
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
