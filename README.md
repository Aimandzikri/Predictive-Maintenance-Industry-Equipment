---

# Predicting Jet Engine Problems Before They Happen

This project is an AI tool that predicts when a jet engine will need maintenance **before** it fails.
It uses real NASA data to help prevent breakdowns, save money, and keep flights safe.

**[Launch Dashboard](https://predictivemaintenance-industry.streamlit.app/)**

**[Watch Demo](https://youtu.be/j2kfLIAS0PY)**

---

## Why This Matters

Most maintenance is done on a schedule or after something breaks.
That’s wasteful and risky.

**Predictive maintenance** uses data to see problems early.
By knowing an engine’s **Remaining Useful Life (RUL)**, you can fix it at the right time.

---

## How It Works

| Step                   | What Happens                                                            | Result                                  |
| ---------------------- | ----------------------------------------------------------------------- | --------------------------------------- |
| 1️⃣ Learn from history | Reads past engine data and finds patterns that lead to failure.         | Knows what “wear and tear” looks like.  |
| 2️⃣ Train the model    | Uses **XGBoost** to connect small sensor changes to remaining lifespan. | A model that can predict future health. |
| 3️⃣ Make predictions   | Runs in an interactive dashboard.                                       | See RUL for any engine at any time.     |

---

## The Model

* **Type:** XGBoost (fast, accurate)
* **Inputs:** 14 sensors + 3 operating settings
  (e.g., temperature, pressure, speed)
* **Method:** Looks at recent sensor trends (last 5 cycles), not just one reading
* **Target:** Remaining Useful Life (in cycles)

---

## Model Accuracy

* **RMSE:** \[Value] cycles → average prediction error
* **MAE:** \[Value] cycles → average difference from actual
* **Within ±30 cycles:** \[Value]% → shows predictions close to real failures

---

## Dashboard Features

1. View accuracy results
2. Pick an engine to inspect
3. Slide through time to see predictions
4. Color-coded RUL gauge:

   * Green: >100 cycles
   * Amber: 30–100 cycles
   * Red: <30 cycles
5. Compare prediction to real RUL
6. See sensor trends
7. View top sensors affecting prediction

---

## Where It Can Be Used

* Real-time monitoring for aircraft or factories
* Automatic alerts when RUL is low
* Overnight fleet analysis

---

<details>
<summary><b>Developer Setup</b></summary>

```bash
# 1. Setup environment (optional)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install packages
pip install -r src/requirements.txt

# 3. Prepare data & train model
python src/preprocess.py
python src/train.py

# 4. Start dashboard
streamlit run src/app.py
```

---

**Folder Structure**

```
Predictive Maintenance/
├── data/          # Data files
├── models/        # Trained model and scaler
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── app.py
│   └── requirements.txt
└── README.md
```

**Dataset:** NASA CMAPSS FD001 (single condition, single fault)
**More info:** [NASA PCoE Data](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

</details>

---
