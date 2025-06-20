"""
Streamlit app for Predictive Maintenance of Turbofan Engines

This app allows users to interact with the trained XGBoost model
for predicting Remaining Useful Life (RUL) of aircraft engines.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import streamlit as st
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import textwrap

# Set page config
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
WINDOW_SIZE = 5  # Must match the window size used in training

# Key sensor columns based on feature importance
KEY_SENSORS = [
    'sensor_14', 'sensor_09', 'sensor_11', 'sensor_13', 
    'sensor_15', 'sensor_04', 'sensor_12', 'sensor_02'
]

# --------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained XGBoost model and related artifacts."""
    try:
        # Load model
        model = xgb.Booster()
        model.load_model(MODEL_DIR / "xgboost_model.json")
        
        # Load scaler
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        
        # Load feature names
        with open(MODEL_DIR / "feature_names.json", 'r') as f:
            feature_names = json.load(f)
        
        # Ensure feature_names is a list
        if not isinstance(feature_names, list):
            feature_names = list(feature_names)
            
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

@st.cache_data
def load_data():
    """Load the processed test data and ground truth RUL."""
    try:
        # Load test data
        test_df = pd.read_csv(DATA_DIR / "test_processed.csv")
        
        # Ensure all necessary columns are present
        expected_cols = [
            'unit_id', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3',
            'sensor_01', 'sensor_02', 'sensor_03', 'sensor_04', 'sensor_05',
            'sensor_06', 'sensor_07', 'sensor_08', 'sensor_09', 'sensor_10',
            'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
            'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
            'sensor_21'
        ]
        
        # Add missing columns with 0 if they don't exist
        for col in expected_cols:
            if col not in test_df.columns and col != 'unit_id' and col != 'time_in_cycles':
                test_df[col] = 0
        
        # Ensure columns are in the correct order
        test_df = test_df[expected_cols].copy()
        
        # Load ground truth RUL
        with open(DATA_DIR.parent / "RUL_FD001.txt", 'r') as f:
            true_rul = [int(line.strip()) for line in f]
        
        # Add true RUL to test data
        test_df['true_rul'] = 0
        for unit_id in test_df['unit_id'].unique():
            test_df.loc[test_df['unit_id'] == unit_id, 'true_rul'] = true_rul[unit_id - 1]
            
        return test_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def compute_benchmarks(test_df: pd.DataFrame, _model, _scaler, feature_names):
    """Compute simple benchmark metrics on the test set (last cycle per engine)."""
    preds, trues = [], []
    for unit_id in test_df['unit_id'].unique():
        engine_data = test_df[test_df['unit_id'] == unit_id]
        current_cycle = engine_data['time_in_cycles'].max()
        dmatrix, _ = prepare_features(engine_data, _scaler, current_cycle)
        if dmatrix is None:
            continue
        pred = float(_model.predict(dmatrix)[0])
        true_val = float(engine_data[engine_data['time_in_cycles'] == current_cycle]['true_rul'].values[0])
        preds.append(pred)
        trues.append(true_val)
    if not preds:
        return None
    rmse = root_mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    within_30 = np.mean(np.abs(np.array(preds) - np.array(trues)) <= 30) * 100
    return {
        'RMSE': rmse,
        'MAE': mae,
        'Accuracy (%)': within_30
    }

def prepare_features(engine_data, scaler, current_cycle):
    """Prepare features for prediction for a specific cycle."""
    # Get historical data up to the current cycle
    history = engine_data[engine_data['time_in_cycles'] <= current_cycle].copy()
    
    # Get all numeric columns except the target and ID columns
    numeric_cols = history.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['unit_id', 'time_in_cycles', 'true_rul']]
    
    # Calculate rolling statistics for all numeric columns
    for col in numeric_cols:
        # Calculate rolling mean and std for each column
        history[f'{col}_rolling_mean'] = history[col].rolling(window=WINDOW_SIZE, min_periods=1).mean()
        history[f'{col}_rolling_std'] = history[col].rolling(window=WINDOW_SIZE, min_periods=1).std().fillna(0)
    
    # Get the most recent row (current cycle)
    if not history.empty:
        # Load expected feature list used during training
        with open(MODEL_DIR / "feature_names.json", 'r') as f:
            expected_cols = json.load(f)

        # Select the most recent row and drop non-feature columns
        current_features = history.iloc[[-1]].copy()
        current_features = current_features.drop(columns=['unit_id', 'time_in_cycles', 'true_rul'], errors='ignore')

        # Ensure all expected columns exist; add any missing with zero
        for col in expected_cols:
            if col not in current_features.columns:
                current_features[col] = 0

        # Keep only expected columns and preserve their order
        current_features = current_features[expected_cols]
        
        # Scale features
        scaled_features = scaler.transform(current_features)
        
        # Convert to DMatrix for prediction - ensure feature_names is a list of strings
        feature_names_list = current_features.columns.tolist()
        dmatrix = xgb.DMatrix(scaled_features, feature_names=feature_names_list)
        
        return dmatrix, current_features
    return None, None

def plot_sensor_trends(engine_data, current_cycle):
    """Plot sensor trends for the selected engine."""
    fig = go.Figure()
    
    # Add vertical line for current cycle
    fig.add_vline(x=current_cycle, line_dash="dash", line_color="red", 
                 annotation_text=f"Current Cycle: {current_cycle}", 
                 annotation_position="top right")
    
    # Plot each key sensor
    for sensor in KEY_SENSORS:
        if sensor in engine_data.columns:
            fig.add_trace(go.Scatter(
                x=engine_data['time_in_cycles'],
                y=engine_data[sensor],
                mode='lines',
                name=sensor,
                opacity=0.7
            ))
    
    # Update layout
    fig.update_layout(
        title="Sensor Trends Over Time",
        xaxis_title="Time in Cycles",
        yaxis_title="Sensor Value",
        legend_title="Sensors",
        hovermode="x",
        height=500
    )
    
    return fig

def main():
    """Main Streamlit app function."""
    st.title("✈️ Predictive Maintenance Dashboard")
    st.caption("Stay ahead of unscheduled maintenance by predicting when each engine will need service.")
    
    # Load model and data
    model, scaler, feature_names = load_model()
    test_df = load_data()
    
    if model is None or test_df is None:
        st.error("Failed to load required data. Please check the logs for errors.")
        return
    
    # Compute and display benchmark metrics for trust
    benchmarks = compute_benchmarks(test_df, model, scaler, feature_names)
    if benchmarks:
        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.metric("Model RMSE", f"{benchmarks['RMSE']:.1f} cycles")
        col_b2.metric("Avg. Abs Error", f"{benchmarks['MAE']:.1f} cycles")
        col_b3.metric("Predictions within ±30 cycles", f"{benchmarks['Accuracy (%)']:.0f}%")
    
    with st.expander("ℹ️  What do these numbers mean?", expanded=False):
        st.markdown(textwrap.dedent("""
        **How we benchmark the model**

        * **RMSE (Root Mean Squared Error):** Think of this as the typical distance between our predictions and reality. Lower is better.
        * **Average Absolute Error:** On average, how many cycles are we off? Again, lower is better.
        * **Predictions within ±30 cycles:** The percentage of engines where our estimate is close (within 30 cycles) to the true value. Higher is better.
        
        These quick stats help you trust that the model is performing reliably on unseen engines.
        """))
    
    # Sidebar for user inputs
    st.sidebar.header("1️⃣  Choose an Engine")
    
    # Get unique engine IDs
    engine_ids = sorted(test_df['unit_id'].unique())
    selected_engine = st.sidebar.selectbox("Select Engine ID", engine_ids)
    
    # Get max cycle for selected engine
    engine_data = test_df[test_df['unit_id'] == selected_engine].copy()
    max_cycle = engine_data['time_in_cycles'].max()
    
    # Slider for cycle selection
    current_cycle = st.sidebar.slider(
        "2️⃣  Select Current Cycle",
        min_value=1,
        max_value=int(max_cycle),
        value=min(50, int(max_cycle)),
        step=1,
        help="Drag to the right as the engine ages."
    )
    
    # Prepare features for prediction
    dmatrix, current_features = prepare_features(engine_data, scaler, current_cycle)
    
    if dmatrix is not None:
        # Make prediction
        predicted_rul = model.predict(dmatrix)[0]
        true_rul = engine_data[engine_data['time_in_cycles'] == current_cycle]['true_rul'].values[0]
        
        # Calculate error
        error = abs(predicted_rul - true_rul)
        
        # Intuitive gauge + metrics
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_rul,
            title={'text': "Predicted RUL (cycles)"},
            gauge={
                'axis': {'range': [0, 125]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "#ff4d4d"},
                    {'range': [30, 100], 'color': "#ffa500"},
                    {'range': [100, 125], 'color': "#2ecc71"}
                ]
            }
        ))
        col1, col2 = st.columns([1,1])
        col1.plotly_chart(gauge_fig, use_container_width=True)
        with col2:
            st.metric("True RUL", f"{true_rul} cycles")
            st.metric("Absolute Error", f"{error:.1f} cycles")
        
        # Health status indicator
        if predicted_rul > 100:
            status = "🟢 Healthy"
            status_color = "green"
        elif predicted_rul > 30:
            status = "🟡 Warning"
            status_color = "orange"
        else:
            status = "🔴 Critical"
            status_color = "red"
            
        st.markdown(f"### Engine Status: <span style='color:{status_color}'>{status}</span>", 
                   unsafe_allow_html=True)
        
        # Plot sensor trends
        st.plotly_chart(plot_sensor_trends(engine_data, current_cycle), use_container_width=True)
        
        # Show feature importance for the current prediction
        st.subheader("Feature Importance for This Prediction")
        
        # Get feature importance scores
        importance = model.get_score(importance_type='weight')
        
        # Convert to DataFrame and sort
        importance_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        }).sort_values('Importance', ascending=False).head(10)
        
        # Plot feature importance
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title='Top 10 Most Important Features',
            labels={'Importance': 'Importance Score', 'Feature': 'Feature'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show raw data for the current cycle
        if st.checkbox("Show Raw Data for Current Cycle"):
            st.subheader("Current Sensor Readings")
            st.dataframe(current_features.T.rename(columns={0: 'Value'}).style.format("{:.4f}"), 
                        use_container_width=True)
    
    # Add some styling
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **How to use:**
        
        1. Select an engine ID from the dropdown
        2. Use the slider to select the current cycle
        3. View the predicted RUL and sensor trends
        
        **Status Indicators:**
        - 🟢 Healthy: RUL > 100 cycles
        - 🟡 Warning: 30 < RUL ≤ 100 cycles
        - 🔴 Critical: RUL ≤ 30 cycles
        """
    )

if __name__ == "__main__":
    main()
