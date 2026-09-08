"""
src/predict.py

Inference pipeline for Pharmacy Inventory Reorder API.
Handles payload parsing, feature engineering, dynamic dummy variable creation, 
and strict feature matrix alignment against trained model metadata.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

_model = None


def load_model():
    """Loads and caches model artifact into memory."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_reorder_quantity(payload) -> Dict[str, Any]:
    """
    Transforms clean user input, performs feature engineering, matches exact model 
    feature names, and generates inference.
    """
    model = load_model()

    # Convert Pydantic payload to DataFrame
    input_df = pd.DataFrame([payload.model_dump()])

    # Feature Engineering
    input_df["days_until_expiry"] = (
        pd.to_datetime(input_df["expiry_date"]) - pd.to_datetime(input_df["transaction_date"])
    ).dt.days
    input_df["days_until_expiry"] = input_df["days_until_expiry"].apply(lambda x: max(1, x))

    input_df["total_amount_pkr"] = input_df["units_dispensed"] * input_df["price_pkr"]
    input_df["inventory_turnover_ratio"] = input_df["units_dispensed"] / (input_df["stock_on_hand"] + 1e-5)
    input_df["expiry_waste_risk_ratio"] = input_df["stock_on_hand"] / (input_df["days_until_expiry"] + 1e-5)

    # Drop raw date fields
    input_df = input_df.drop(columns=["transaction_date", "expiry_date"])

    # Categorical One-Hot Encoding
    encoded_df = pd.get_dummies(input_df)

    # Strict Feature Matrix Alignment
    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is not None:
        final_df = encoded_df.reindex(columns=expected_features, fill_value=0)
    else:
        final_df = encoded_df

    # Predict & Post-process
    raw_pred = model.predict(final_df)[0]
    recommended_units = max(0, int(round(raw_pred)))

    waste_risk = final_df["expiry_waste_risk_ratio"].iloc[0]
    risk_status = "HIGH SPOILAGE RISK" if waste_risk > 1.5 else "STANDARD REORDER"

    return {
        "status": "success",
        "recommended_reorder_units": recommended_units,
        "spoilage_risk_status": risk_status,
        "engineered_metrics": {
            "days_until_expiry": int(final_df["days_until_expiry"].iloc[0]),
            "total_amount_pkr": float(round(final_df["total_amount_pkr"].iloc[0], 2)),
            "inventory_turnover_ratio": float(round(final_df["inventory_turnover_ratio"].iloc[0], 4)),
            "expiry_waste_risk_ratio": float(round(final_df["expiry_waste_risk_ratio"].iloc[0], 4)),
        },
    }