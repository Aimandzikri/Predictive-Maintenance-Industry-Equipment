"""
Model training script for Predictive Maintenance project.

This script performs feature engineering (rolling window features),
trains an XGBoost regression model, and saves the model artifacts.
"""
import os
import sys
import json
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
WINDOW_SIZE = 5  # Number of cycles to include in rolling window
FEATURES_TO_SCALE = [
    'op_setting_1', 'op_setting_2',
    'sensor_02', 'sensor_03', 'sensor_04', 'sensor_05',
    'sensor_06', 'sensor_07', 'sensor_08', 'sensor_09',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
    'sensor_15', 'sensor_16', 'sensor_17', 'sensor_20', 'sensor_21'
]
ROLLING_STATS = ['mean', 'std']

# Paths
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load preprocessed data."""
    try:
        logger.info("Loading preprocessed data...")
        train_path = DATA_DIR / "train_processed.csv"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
            
        train_df = pd.read_csv(train_path)
        logger.info(f"Loaded data shape: {train_df.shape}")
        
        if 'RUL' not in train_df.columns:
            raise ValueError("RUL column not found in training data")
            
        # Separate features and target
        X = train_df.drop(columns=['RUL'])
        y = train_df['RUL']
        
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_rolling_features(df: pd.DataFrame, y: pd.Series = None, window: int = WINDOW_SIZE) -> Tuple[pd.DataFrame, pd.Series]:
    """Create rolling window features for each engine.
    
    Args:
        df: Input DataFrame with sensor readings
        y: Optional target series to keep aligned with features
        window: Size of the rolling window
        
    Returns:
        Tuple of (features_df, y_aligned) where y_aligned is None if y was None
    """
    logger.info(f"Creating rolling window features (window={window})...")
    
    # Make copies to avoid SettingWithCopyWarning
    df = df.copy()
    if y is not None:
        y = y.copy()
    
    # Only keep columns we need for rolling features
    rolling_cols = ['unit_id', 'time_in_cycles'] + [col for col in FEATURES_TO_SCALE if col in df.columns]
    df = df[rolling_cols].copy()
    
    # Sort by unit_id and time_in_cycles to ensure proper rolling
    sort_cols = ['unit_id', 'time_in_cycles']
    df = df.sort_values(sort_cols)
    
    # Initialize lists to store results
    results = []
    y_results = [] if y is not None else None
    
    # Process each engine separately
    for unit_id in df['unit_id'].unique():
        # Get the group for this unit
        group = df[df['unit_id'] == unit_id].copy()
        
        # Get the corresponding y values if provided
        if y is not None:
            y_group = y[df['unit_id'] == unit_id].iloc[window-1:].copy()
        
        # Calculate rolling statistics for each sensor
        for sensor in FEATURES_TO_SCALE:
            if sensor not in group.columns:
                continue
                
            # Rolling mean
            group[f'{sensor}_rolling_mean'] = group[sensor].rolling(
                window=window, min_periods=1).mean()
                
            # Rolling standard deviation
            group[f'{sensor}_rolling_std'] = group[sensor].rolling(
                window=window, min_periods=1).std().fillna(0)
        
        # Remove the first (window-1) rows that don't have full window
        group = group.iloc[window-1:]
        results.append(group)
        
        # Store the aligned y values
        if y is not None:
            y_results.append(y_group)
    
    # Combine all engines
    result_df = pd.concat(results)
    
    # Combine y values if they were provided
    y_aligned = pd.concat(y_results) if y_results is not None else None
    
    # Sort the final dataframes
    result_df = result_df.sort_values(sort_cols)
    if y_aligned is not None:
        y_aligned = y_aligned.sort_index()
    
    return result_df, y_aligned

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, Any] = None
) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """Train XGBoost regression model.
    
    Args:
        X: Feature matrix
        y: Target values
        params: XGBoost parameters
        
    Returns:
        Tuple of (trained model, feature importance)
    """
    print("Training XGBoost model...")
    
    # Default parameters if none provided
    if params is None:
        params = {
            'n_estimators': 100,  # Reduced from 200
            'max_depth': 4,       # Reduced from 5
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE,
            'eval_metric': 'rmse',
            'early_stopping_rounds': 10,
            'tree_method': 'hist',  # More memory efficient
            'n_jobs': -1,           # Use all cores
        }
    
    # Convert to DMatrix for better memory efficiency
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=False)
    
    # Train model with progress tracking
    print("Training progress:")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        verbose_eval=10,  # Print progress every 10 iterations
    )
    
    # Get feature importance
    importance = model.get_score(importance_type='weight')
    
    # Convert to sklearn interface for compatibility
    sklearn_model = xgb.XGBRegressor(**params)
    sklearn_model._Booster = model
    
    return sklearn_model, importance

def evaluate_model(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    split: str = 'train'
) -> float:
    """Evaluate model performance.
    
    Args:
        model: Trained XGBoost model
        X: Feature matrix
        y: True target values
        split: Name of the split being evaluated (for logging)
        
    Returns:
        RMSE score
    """
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"{split.capitalize()} RMSE: {rmse:.2f}")
    return rmse

def save_artifacts(
    model: xgb.XGBRegressor,
    scaler: MinMaxScaler,
    feature_importance: Dict[str, float],
    feature_names: List[str]
) -> None:
    """Save model artifacts to disk."""
    print("Saving model artifacts...")
    
    # Save model
    model.save_model(MODEL_DIR / "xgboost_model.json")
    
    # Save scaler
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    
    # Save feature importance
    with open(MODEL_DIR / "feature_importance.json", 'w') as f:
        json.dump(feature_importance, f, indent=2)
    
    # Save feature names
    with open(MODEL_DIR / "feature_names.json", 'w') as f:
        json.dump(feature_names, f, indent=2)

def log_memory_usage():
    """Log current memory usage."""
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory usage: {mem_info.rss / (1024 * 1024):.2f} MB")

def main():
    """Main training pipeline."""
    start_time = datetime.now()
    logger.info(f"Starting model training at {start_time}")
    
    try:
        # 1. Load data
        logger.info("\n[1/5] Loading data...")
        X, y = load_data()
        logger.info(f"Loaded {len(X)} samples with {len(X.columns)} features")
        log_memory_usage()
        
        # 2. Create rolling window features and align target
        logger.info("\n[2/5] Creating features...")
        X_engineered, y_aligned = create_rolling_features(X, y)
        logger.info(f"Created {len(X_engineered.columns)} features")
        logger.info(f"X_engineered shape: {X_engineered.shape}, y_aligned shape: {y_aligned.shape}")
        log_memory_usage()
        
        # 3. Scale features
        logger.info("\n[3/5] Scaling features...")
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_engineered)
        
        # Convert back to DataFrame for feature names
        X_scaled = pd.DataFrame(X_scaled, columns=X_engineered.columns)
        logger.info(f"Scaled data shape: {X_scaled.shape}")
        log_memory_usage()
        
        # 4. Train model
        logger.info("\n[4/5] Training model...")
        model, feature_importance = train_model(X_scaled, y_aligned)
        log_memory_usage()
        
        # 5. Evaluate on training set
        logger.info("\n[5/5] Evaluating model...")
        rmse = evaluate_model(model, X_scaled, y_aligned, 'train')
        logger.info(f"Training RMSE: {rmse:.4f}")
        
        # 6. Save artifacts
        logger.info("\nSaving artifacts...")
        save_artifacts(
            model=model,
            scaler=scaler,
            feature_importance=feature_importance,
            feature_names=X_engineered.columns.tolist()
        )
        
        # Log completion
        duration = (datetime.now() - start_time).total_seconds() / 60
        logger.info(f"\nTraining complete in {duration:.2f} minutes!")
        logger.info(f"Model artifacts saved to: {MODEL_DIR.absolute()}")
        
        # Print top features
        logger.info("\nTop 10 most important features:")
        for i, (feat, imp) in enumerate(sorted(feature_importance.items(), 
                                             key=lambda x: x[1], 
                                             reverse=True)[:10], 1):
            logger.info(f"{i}. {feat}: {imp:.2f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        return 1
    finally:
        logger.info("\n" + "="*50 + "\n")

if __name__ == "__main__":
    try:
        import psutil
    except ImportError:
        logger.info("Installing psutil for memory monitoring...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    
    logger.info("="*50)
    logger.info("Starting new training session")
    logger.info("="*50)
    
    sys.exit(main())
