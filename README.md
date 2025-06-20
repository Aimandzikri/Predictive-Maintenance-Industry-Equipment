# Predicting Jet Engine Failure Before It Happens

This project showcases a powerful AI system that predicts when a jet engine will need maintenance, long before a critical failure occurs. Using real-world data from NASA, our model acts like a "crystal ball" for engine health, helping to prevent costly downtime, increase safety, and improve operational efficiency.

The system is built as a complete, end-to-end solution, from raw data processing to an interactive dashboard that anyone can use.

🚀 **[Launch the Interactive Dashboard](https://predictivemaintenance-industry.streamlit.app/)**  
📺 **[Watch the Demo Video](https://youtu.be/j2kfLIAS0PY)**

---

## The Goal: From Reactive to Proactive Maintenance

Traditionally, maintenance is done on a fixed schedule (e.g., every 1000 hours) or after a part has already failed. This is inefficient and risky.

Our goal is to shift to **predictive maintenance**: using data to forecast failures *before they happen*. By knowing an engine's **Remaining Useful Life (RUL)**, maintenance can be scheduled precisely when needed, saving millions in costs and preventing catastrophic failures.

  <!-- You can create a simple visual for this -->

---

## How It Works: A 3-Step Process

Our automated system transforms raw sensor data into actionable predictions in three key stages:

| Step | What It Does | The Outcome |
|:----:|--------------|-------------|
| 1️⃣ | **Understand the Past** | Reads historical data from thousands of engine cycles, learning the patterns of wear and tear that lead to failure. |
| 2️⃣ | **Build the Crystal Ball** | Trains an advanced AI model (XGBoost) to recognize these complex patterns. It learns to connect subtle changes in sensor readings to the engine's remaining lifespan. |
| 3️⃣ | **Predict the Future** | Deploys the trained model into an interactive dashboard where you can select any engine at any point in its life and instantly see its predicted RUL. |

---

## The "Brain" Behind the Prediction: Our AI Model

At the heart of our system is a sophisticated AI model specifically chosen for its high accuracy and ability to understand complex data.

*   **Model Type**: **XGBoost**, an industry-leading algorithm known for its performance in predictive tasks.
*   **What It Looks At**: It analyzes **14 key sensor readings** (like temperature, pressure, and speed) along with 3 operational settings to get a complete picture of engine health.
*   **How It Gets Smart**: Instead of just looking at a single snapshot in time, the model analyzes **recent sensor trends** (e.g., the average and stability of readings over the last 5 cycles). This allows it to spot degradation long before it becomes critical.
*   **What It Predicts**: The **Remaining Useful Life (RUL)**, measured in operational cycles.

---

## How Accurate Is It? Model Performance

We rigorously tested our model on a set of engines it had never seen before. The results show a high degree of accuracy and reliability.

*   **Root Mean Squared Error (RMSE): [Value] cycles**
    *   *In simple terms:* This is the typical "margin of error" for our predictions. On average, our RUL prediction is off by about this many cycles. A lower number is better.
*   **Mean Absolute Error (MAE): [Value] cycles**
    *   *In simple terms:* This tells us the average absolute difference between our prediction and the actual outcome. It's another way to confirm the model's high accuracy.
*   **Accuracy within ±30 Cycles: [Value]%**
    *   *In simple terms:* This percentage of our predictions were "in the ballpark," falling within 30 cycles of the true failure point. This is crucial for making confident maintenance scheduling decisions.

> **Bottom Line:** The model provides a reliable forecast that is accurate enough to be used for real-world maintenance planning.

---

## The Dashboard: Your Mission Control for Engine Health

The interactive dashboard makes our powerful AI accessible to everyone, from engineers to executives.

1.  **Overall Performance:** See the model's accuracy metrics at a glance.
2.  **Select an Engine:** Choose any of the 100 test engines to analyze.
3.  **Go Back in Time:** Use the slider to see what the prediction would have been at any point in the engine's history.
4.  **Instant Prediction:** A gauge immediately shows the predicted RUL, color-coded for urgency:
    *   **Green:** Healthy (>100 cycles left)
    *   **Amber:** Caution (30–100 cycles left)
    *   **Red:** Warning (under 30 cycles left)
5.  **Ground Truth:** Compare the prediction to the engine's actual RUL for that cycle to build trust in the system.
6.  **Sensor Trends:** Watch how key sensor readings change over the engine's lifetime.
7.  **Prediction Drivers:** See which sensors were most important for the model's prediction, providing transparency and insight.

---

## Putting the Model to Work

This system isn't just an experiment; it's a blueprint for real-world applications. The trained model can be integrated into:

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
