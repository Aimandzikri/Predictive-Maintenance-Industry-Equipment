# Predictive Maintenance for Turbofan Engines (NASA CMAPSS FD001)

This project demonstrates an end-to-end **Remaining Useful Life (RUL)** prediction pipeline for aircraft engines using the NASA CMAPSS dataset (single-fault scenario **FD001**).

The repository is organised around three production-ready Python scripts – **no notebooks** – and a Streamlit dashboard for interactive exploration.

---

## 1. Quick Start

```bash
# 1. (Optional) create & activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r src/requirements.txt

# 3. Pre-process data & train model (≈ 1 minute)
python src/preprocess.py
python src/train.py

# 4. Launch the dashboard
streamlit run src/app.py
```

Open the URL printed by Streamlit to view the dashboard.

---

## 2. Repository Structure

```
Predictive Maintenance for Industrial Equipment/
├── data/               # Raw & processed data (git-ignored except small samples)
├── models/             # Saved model, scaler & metadata (after training)
├── src/
│   ├── preprocess.py   # Data cleaning & RUL engineering
│   ├── train.py        # Feature engineering & XGBoost training
│   ├── app.py          # Streamlit dashboard
│   └── requirements.txt
└── README.md
```

---

## 3. Model Architecture & Training

The predictive maintenance system is powered by an **XGBoost** (eXtreme Gradient Boosting) regressor, chosen for its superior performance in handling time-series sensor data and its ability to capture complex, non-linear relationships in the data.

### Key Model Features:
- **Algorithm**: XGBoost Regressor with optimized hyperparameters
- **Input Features**: 14 sensor readings + 3 operational settings
- **Feature Engineering**:
  - Rolling window statistics (5-cycle window)
  - Mean and standard deviation calculations
  - Min-Max scaling for sensor normalization
- **Target Variable**: Remaining Useful Life (RUL) in cycles
- **Training Objective**: Minimize Root Mean Squared Error (RMSE)
- **Model Artifacts**:
  - `xgboost_model.json`: Serialized model weights
  - `scaler.pkl`: Fitted MinMaxScaler for data normalization
  - `feature_names.json`: List of feature names for inference

### Model Performance:
- **RMSE**: [Value] cycles (on test set)
- **MAE**: [Value] cycles
- **±30 cycles accuracy**: [Value]%

## 4. Pipeline Overview

| Step | Script | Key Actions |
|------|--------|-------------|
| 1️⃣ | `preprocess.py` | • Load **train_FD001.txt** / **test_FD001.txt**  \<br>• Remove constant sensors  \<br>• Compute & cap RUL at 125 cycles  \<br>• Save `train_processed.csv`, `test_processed.csv` |
| 2️⃣ | `train.py` | • Feature engineering with rolling windows (mean & std, window=5)  \<br>• Min-Max scaling for sensor normalization  \<br>• Train **XGBoost** regressor with early stopping  \<br>• Save model artifacts (`xgboost_model.json`, `scaler.pkl`, `feature_names.json`) |
| 3️⃣ | `app.py` | • Interactive dashboard for model inference  \<br>• Performance metrics (RMSE, MAE, % within ±30 cycles)  \<br>• Real-time RUL predictions for individual engines  \<br>• Sensor trend visualization and feature importance analysis |

---

## 5. Dashboard Walk-Through

1. **Model Benchmarks** – Quickly assess model quality on unseen engines:  
   • **RMSE** – Typical prediction error (cycles)  
   • **Avg. Absolute Error** – Mean absolute deviation  
   • **% within ±30 cycles** – Fraction of engines where prediction is close.
2. **Choose an Engine** – Select an engine ID (1-100).
3. **Select Current Cycle** – Drag the slider to any historical cycle.
4. **Predicted RUL Gauge** – Immediate visual of remaining life:  
   • Green (>100 cycles), Amber (30–100), Red (≤30).
5. **True RUL & Error** – Ground-truth comparison for trust.
6. **Sensor Trends** – Line chart of key sensors over time with the current cycle highlighted.
7. **Feature Importance** – Top model drivers for transparency.

> **Tip:** Use the expander below the benchmarks for a plain-English explanation of each metric.

---

## 6. Model Explainability & Interpretability

The model's predictions are made interpretable through:
- **Feature Importance**: Visualizing which sensor readings most influence predictions
- **SHAP Values**: Explaining individual predictions by showing the contribution of each feature
- **Trend Analysis**: Highlighting how sensor behavior evolves as the engine approaches failure

## 7. Model Deployment & Inference

The trained model can be easily integrated into production systems through:
- **REST API** (Flask/FastAPI wrapper)
- **Batch Processing** for offline predictions
- **Real-time Monitoring** for live equipment tracking

---

## 8. Dataset

NASA Turbofan Engine Degradation Simulation (CMAPSS) – subset **FD001** (single operating condition, single fault). Original data available at the [PHM Society Data Challenge](https://www.phmsociety.org/).

---

## 9. Requirements

Python ≥3.9. All Python package versions are pinned in `src/requirements.txt`.

---
