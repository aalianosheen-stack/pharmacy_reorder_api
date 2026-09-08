"""
main.py

FastAPI REST API Service for Pharmacy Reorder Prediction.
"""

from datetime import date
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import predict_reorder_quantity

app = FastAPI(
    title="Hospital Pharmacy Reorder System API",
    description="REST API for predicting medicine reorder quantities and evaluating waste risk.",
    version="1.0.0",
)


class PharmacyInputSchema(BaseModel):
    week: int = Field(..., ge=1, le=52, example=12)
    units_dispensed: float = Field(..., ge=0, example=150.0)
    stock_on_hand: float = Field(..., ge=0, example=45.0)
    units_ordered: float = Field(..., ge=0, example=50.0)
    price_pkr: float = Field(..., gt=0, example=350.0)
    stockout: float = Field(0.0, ge=0, le=1, example=0.0)

    patient_age_group: str = Field("Adult", example="Adult")
    copay_type: str = Field("Cash", example="Cash")
    prescriber_specialty: str = Field("General Physician", example="General Physician")

    transaction_date: date = Field(default_factory=date.today, example="2026-09-08")
    expiry_date: date = Field(..., example="2026-10-15")


@app.get("/", tags=["Health Check"])
def health_check() -> Dict[str, str]:
    return {"status": "online", "service": "Pharmacy Reorder API"}


@app.post("/predict", tags=["Inference"])
def predict_endpoint(payload: PharmacyInputSchema) -> Dict[str, Any]:
    try:
        return predict_reorder_quantity(payload)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))