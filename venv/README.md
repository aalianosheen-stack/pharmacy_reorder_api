# 💊 Hospital Pharmacy Reorder System — REST API

An enterprise-ready **FastAPI REST Service** designed to predict optimal stock reorder quantities and mitigate medicine spoilage risk in hospital pharmacy operations.

---

## 🎯 Overview

Managing pharmacy inventory requires balancing stock availability against waste from batch expirations. This microservice exposes a lightweight machine learning endpoint that accepts transaction signals (dispense velocity, stock on hand, expiry windows) and generates real-time inventory recommendations.

---

## 🏗️ Architecture & Project Structure

```text
pharmacy_reorder_api/
│
├── .gitignore
├── README.md
├── requirements.txt
├── main.py                    # FastAPI application & Pydantic schema
│
├── models/
│   └── best_model.pkl         # Serialized Scikit-Learn Model
│
└── src/
    ├── __init__.py
    └── predict.py             # Inference pipeline & feature engineering