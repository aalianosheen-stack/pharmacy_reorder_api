# 💊 Hospital Pharmacy Reorder System — REST API

An end-to-end AI-powered inventory demand forecasting and expiry waste mitigation backend system built with **FastAPI**, **Pydantic**, and **Scikit-Learn**. 

This REST API microservice processes real-time dispensing telemetry and stock metrics to generate low-stock reorder warnings, optimal reorder quantities, and expiry risk alerts for hospital inventory management.

---

## 🎯 Key Features

- **RESTful Endpoints**: Built with FastAPI for high-throughput inference and automatic OpenAPI/Swagger documentation.
- **Strict Data Validation**: Pydantic schemas enforce type safety, value ranges, and missing-field checking on incoming payloads.
- **Machine Learning Inference**: Serialized Scikit-Learn pipelines predict demand metrics and flag reorder thresholds.
- **Modular Microservice Architecture**: Clean separation between API routing (`main.py`) and inference logic (`src/predict.py`).

---

## 🏗️ Architecture & Project Structure

```text
pharmacy_reorder_api/
├── .gitignore
├── README.md
├── requirements.txt
├── main.py                # FastAPI application & Pydantic request/response schemas
├── models/
│   └── best_model.pkl    # Serialized Scikit-Learn Model Pipeline
└── src/
    ├── __init__.py
    └── predict.py         # Feature engineering & ML prediction pipeline