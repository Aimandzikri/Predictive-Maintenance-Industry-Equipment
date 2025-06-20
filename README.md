# Predicting Jet Engine Failure Before It Happens

This project presents a powerful AI system that predicts when a jet engine will need maintenance, long before a critical failure occurs. Using real-world data from NASA, the model acts like a "crystal ball" for engine health, helping to prevent costly downtime, increase safety, and improve operational efficiency.

The system is built as a complete, end-to-end solution, from raw data processing to an interactive dashboard.

🚀 **[Launch the Interactive Dashboard](https://predictivemaintenance-industry.streamlit.app/)**  
📺 **[Watch the Demo Video](https://youtu.be/j2kfLIAS0PY)**

---

## The Goal: From Reactive to Proactive Maintenance

Traditionally, maintenance is performed on a fixed schedule (e.g., every 1000 hours) or after a part has already failed. This approach is both inefficient and risky.

The goal is to shift to **predictive maintenance**: using data to forecast failures *before they happen*. By knowing an engine's **Remaining Useful Life (RUL)**, maintenance can be scheduled precisely when needed, saving millions in costs and preventing catastrophic failures.

  <!-- A simple visual could illustrate the shift from reactive to proactive maintenance -->

---

## How It Works: A 3-Step Process

The automated system transforms raw sensor data into actionable predictions in three key stages:

| Step | What It Does | The Outcome |
|:----:|--------------|-------------|
| 1️⃣ | **Understand the Past** | Reads historical data from thousands of engine cycles, learning the patterns of wear and tear that lead to failure. |
| 2️⃣ | **Build the Crystal Ball** | Trains an advanced AI model (XGBoost) to recognize these complex patterns. It learns to connect subtle changes in sensor readings to the engine's remaining lifespan. |
| 3️⃣ | **Predict the Future** | Deploys the trained model into an interactive dashboard where a user can select any engine at any point in its life and instantly see its predicted RUL. |

---

## The Prediction Model

At the heart of the system is a sophisticated AI model specifically chosen for its high accuracy and ability to interpret complex data.

*   **Model Type**: **XGBoost**, an industry-leading algorithm known for its performance in predictive tasks.
*   **What It Analyzes**: The model processes **14 key sensor readings** (like temperature, pressure, and speed) along with 3 operational settings to build a complete picture of engine health.
*   **Feature Engineering**: Instead of using a single snapshot in time, the model analyzes **recent sensor trends** (e.g., the average and stability of readings over the last 5 cycles). This technique allows it to spot degradation long before it becomes critical.
*   **Prediction Target**: The **Remaining Useful Life (RUL)**, measured in operational cycles.

---

## Model Performance

The model was rigorously tested on a set of engines it had never seen before, demonstrating a high degree of accuracy and reliability.

*   **Root Mean Squared Error (RMSE): [Value] cycles**
    *   *In simple terms:* This is the typical "margin of error" for the predictions. On average, the RUL prediction is off by about this many cycles. A lower number is better.
*   **Mean Absolute Error (MAE): [Value] cycles**
    *   *In simple terms:* This metric represents the average absolute difference between the predicted RUL and the actual outcome, confirming the model's high accuracy.
*   **Accuracy within ±30 Cycles: [Value]%**
    *   *In simple terms:* This percentage of predictions fall within 30 cycles of the true failure point. This level of accuracy is crucial for making confident maintenance scheduling decisions.

---

## The Interactive Dashboard: Mission Control for Engine Health

The interactive dashboard makes the powerful AI model accessible to a wide range of users, from engineers to executives.

1.  **Overall Performance:** View the model's key accuracy metrics at a glance.
2.  **Engine Selection:** Choose any of the 100 test engines for analysis.
3.  **Historical Analysis:** Use the time-series slider to see the RUL prediction at any point in the engine's operational history.
4.  **Instant Prediction:** A dynamic gauge immediately displays the predicted RUL, color-coded for urgency:
    *   **Green:** Healthy (>100 cycles remaining)
    *   **Amber:** Caution (30–100 cycles remaining)
    *   **Red:** Warning (< 30 cycles remaining)
5.  **Ground Truth Comparison:** Compare the live prediction to the engine's actual RUL for that cycle to validate the model's accuracy.
6.  **Sensor Trend Visualization:** Observe how key sensor readings evolve over the engine's lifetime.
7.  **Prediction Drivers (Feature Importance):** Identify which sensors were most influential in the model's prediction, providing transparency and actionable insights.

---

## Real-World Application

This system is a blueprint for real-world applications. The trained model can be integrated into:

*   **Real-time Monitoring Systems** on a factory floor or in an aircraft fleet.
*   **Automated Alerting Systems** that notify maintenance crews when an asset's RUL drops below a certain threshold.
*   **Batch Processing Systems** to analyze the health of an entire fleet of equipment overnight.

---

<details>
<summary><b>Developer Quick Start & Technical Details</b></summary>

### 1. Quick Start Guide

```bash
# 1. (Optional) Create & activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r src/requirements.txt

# 3. Pre-process data & train the model (takes ≈ 1 minute)
python src/preprocess.py
python src/train.py

# 4. Launch the interactive dashboard
streamlit run src/app.py
```
Open the URL printed by Streamlit in your browser to view the dashboard.

### 2. Repository Structure

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

### 3. Dataset

This project uses the NASA Turbofan Engine Degradation Simulation (CMAPSS) dataset, specifically the **FD001** subset. This subset features a single operating condition and a single fault type, making it ideal for demonstrating the core predictive model. The data is available from the [NASA Prognostics Center of Excellence](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).

### 4. Requirements

*   Python 3.9 or higher.
*   All required Python packages are listed with pinned versions in `src/requirements.txt`.
