"""
Data preprocessing script for Predictive Maintenance project.

This script loads the NASA CMAPSS dataset, performs data cleaning, calculates RUL,
and saves the processed data for model training and testing.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

# Define constants
RUL_CAP = 125  # Cap RUL at this value as per project requirements
DATA_DIR = Path("data")
PROCESSED_DIR = Path("data/processed")

# Define column names for the dataset
COLUMNS = [
    'unit_id', 'time_in_cycles',
    'op_setting_1', 'op_setting_2', 'op_setting_3'
] + [f'sensor_{i:02d}' for i in range(1, 22)]  # sensor_01 to sensor_21

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the raw data.
    
    Returns:
        Tuple containing (train_df, test_df) DataFrames
    """
    print("Loading raw data...")
    
    # Load training and test data
    train_path = DATA_DIR / "train_FD001.txt"
    test_path = DATA_DIR / "test_FD001.txt"
    
    train_df = pd.read_csv(train_path, sep="\s+", header=None, names=COLUMNS)
    test_df = pd.read_csv(test_path, sep="\s+", header=None, names=COLUMNS)
    
    return train_df, test_df

def calculate_rul(df: pd.DataFrame, rul_cap: int = None) -> pd.Series:
    """Calculate Remaining Useful Life (RUL) for each row.
    
    Args:
        df: Input DataFrame with 'unit_id' and 'time_in_cycles' columns
        rul_cap: Optional cap for RUL values
        
    Returns:
        Series with RUL values
    """
    # Group by unit and find the max cycle for each unit
    max_cycle = df.groupby('unit_id')['time_in_cycles'].max().reset_index()
    max_cycle.columns = ['unit_id', 'max_cycle']
    
    # Merge to get max cycle for each row and calculate RUL
    df = df.merge(max_cycle, on='unit_id', how='left')
    rul = (df['max_cycle'] - df['time_in_cycles']).astype(int)
    
    # Apply RUL cap if specified
    if rul_cap is not None:
        rul = rul.clip(upper=rul_cap)
    
    return rul

def remove_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Remove constant value columns from the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (filtered DataFrame, list of removed columns)
    """
    # Calculate standard deviation for each column
    std_dev = df.std()
    
    # Find columns with zero standard deviation (constant values)
    constant_columns = std_dev[std_dev == 0].index.tolist()
    
    # Remove constant columns
    filtered_df = df.drop(columns=constant_columns)
    
    return filtered_df, constant_columns

def preprocess_data() -> None:
    """Main function to preprocess the data."""
    print("Starting data preprocessing...")
    
    # Create processed directory if it doesn't exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_df, test_df = load_data()
    
    # Calculate RUL for training data
    print("Calculating RUL for training data...")
    train_df['RUL'] = calculate_rul(train_df, rul_cap=RUL_CAP)
    
    # For test data, we'll keep the last cycle for each unit for evaluation
    # The actual RUL will be loaded from RUL_FD001.txt later
    
    # Remove constant columns
    print("Removing constant columns...")
    train_df, removed_cols = remove_constant_columns(train_df)
    test_df = test_df.drop(columns=removed_cols, errors='ignore')
    
    # Save processed data
    print("Saving processed data...")
    train_df.to_csv(PROCESSED_DIR / "train_processed.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test_processed.csv", index=False)
    
    print(f"Preprocessing complete. Removed constant columns: {removed_cols}")
    print(f"Processed data saved to: {PROCESSED_DIR.absolute()}")

if __name__ == "__main__":
    preprocess_data()
