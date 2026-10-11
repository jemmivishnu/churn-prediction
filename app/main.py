"""FastAPI service that serves churn predictions from the trained model."""

import json
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
METADATA_PATH = MODEL_PATH.with_name("metadata.json")

YesNo = Literal["Yes", "No"]
InternetOption = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
                "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
                "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 191.0,
            }
        }
    }


class Prediction(BaseModel):
    churn_probability: float
    prediction: Literal["Churn", "No Churn"]
    risk_level: Literal["Low", "Medium", "High"]


model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Predicts whether a telecom customer is likely to churn.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Churn Prediction API. Open /docs to try it."}


@app.get("/health")
def health():
    return {"status": "ok", "model": metadata.get("model_type"), "metrics": metadata.get("metrics")}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(df)[0, 1])
    risk = "High" if probability >= 0.7 else "Medium" if probability >= 0.4 else "Low"
    return Prediction(
        churn_probability=round(probability, 4),
        prediction="Churn" if probability >= 0.5 else "No Churn",
        risk_level=risk,
    )
