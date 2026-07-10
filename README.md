
An AI model which is trained on the NASA's CMAPSS dataset for prediction of RUL of Turbofan blades in jet engines, etc.
# Predictive Maintenance System - NASA C-MAPSS FD001

An AI-powered predictive maintenance system for turbofan engines, trained on the **NASA C-MAPSS FD001** dataset. Predicts **Remaining Useful Life (RUL)** of jet engines using XGBoost with rolling window features.



## Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**
- **FD001**: 100 training engines, 100 test engines
- **Degradation mode**: High Pressure Compressor (HPC) degradation
- **Operating conditions**: 6 flight conditions, 21 sensor readings

## Model Performance

| Metric | Score |
|--------|-------|
| RMSE | 19.44 cycles |
| MAE | 13.86 cycles |
| NASA Asymmetric Score | 1338.5 |

## Features

- **51 engineered features**: 17 base sensors/settings + rolling mean & std (window=10)
- **XGBoost regression** with early stopping
- **Real-time RUL prediction** via REST API
- **Interactive dashboard** with NASA aerospace theme
- **Batch analysis** for multiple engines

## Project Structure

```
predictive-maintenance/
├── config/
│   └── settings.py              # App configuration
├── data/
│   ├── train_FD001.txt          # NASA C-MAPSS training data
│   ├── test_FD001.txt           # NASA C-MAPSS test data
│   ├── RUL_FD001.txt            # Actual RUL for test engines
│   ├── batch_analysis_sample.csv
│   ├── batch_low_rul.csv        # Test: engines near failure
│   └── batch_high_rul.csv       # Test: engines in good health
├── src/
│   ├── data/
│   │   └── cmapss_loader.py     # Data loading & feature engineering
│   ├── inference/
│   │   └── cmapss_predictor.py  # Model prediction service
│   └── api/
│       ├── main.py              # FastAPI application
│       └── schemas/
│           ├── request.py       # Request schemas
│           └── response.py      # Response schemas
├── models/
│   ├── xgboost_rul_model.json   # Trained XGBoost model
│   ├── model_metadata.json      # Model info & metrics
│   └── feature_columns.json     # Feature names
├── dashboard/
│   └── streamlit_app.py         # Interactive dashboard
├── scripts/
│   ├── train_cmapss.py          # Model training script
│   └── generate_sample_data.py  # Sample data generator
└── assets/
    └── jet_engine.png           # Dashboard background
```

## Installation

```bash
git clone https://github.com/your-org/predictive-maintenance.git
cd predictive-maintenance
pip install pandas numpy scikit-learn xgboost fastapi uvicorn pydantic pydantic-settings python-dotenv websockets streamlit plotly requests
```

## Quick Start

### 1. Train Model (or use pre-trained)

### 2. Start API Server

### 3. Start Dashboard


## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/model/info` | Model metadata |
| POST | `/api/v1/predict/rul` | Predict RUL for single reading |

## Dashboard Features

- **Predict RUL**: Adjust sensor sliders to get real-time predictions
- **Batch Analysis**: Upload CSV for multiple engine predictions
- **Model Info**: View model metrics and feature importance
- **Sensor Reference**: Detailed sensor descriptions

## Tech Stack

- **ML**: XGBoost, Pandas, NumPy, Scikit-learn
- **API**: FastAPI, Uvicorn, Pydantic
- **Dashboard**: Streamlit, Plotly
- **Dataset**: NASA C-MAPSS FD001
