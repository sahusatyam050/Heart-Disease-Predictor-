"""
Data Preprocessing Utilities for Heart Disease Predictor

This module contains functions for cleaning, transforming, and preparing
the heart disease dataset for machine learning.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """
    Load heart disease dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None


def check_missing_values(df):
    """
    Check for missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.Series: Count of missing values per column
    """
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅ No missing values found")
    else:
        print("⚠️ Missing values detected:")
        print(missing[missing > 0])
    return missing


def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled
    """
    df_clean = df.copy()
    
    if strategy == 'drop':
        df_clean = df_clean.dropna()
    elif strategy in ['mean', 'median']:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if strategy == 'mean':
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        else:
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    
    print(f"✅ Missing values handled using '{strategy}' strategy")
    return df_clean


def prepare_features(df, target_column='target'):
    """
    Separate features (X) and target (y) from the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        
    Returns:
        tuple: (X, y) - Features and target
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"✅ Features: {X.shape[1]} columns")
    print(f"✅ Target: {target_column}")
    
    return X, y


def scale_features(X_train, X_test=None):
    """
    Scale features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame, optional): Test features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler) or (X_train_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        print("✅ Features scaled successfully (train and test)")
        return X_train_scaled, X_test_scaled, scaler
    
    print("✅ Features scaled successfully (train only)")
    return X_train_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X: Features
        y: Target
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✅ Data split: {len(X_train)} training samples, {len(X_test)} test samples")
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(filepath, target_column='target', test_size=0.2):
    """
    Complete preprocessing pipeline from loading to scaled train/test split.
    
    Args:
        filepath (str): Path to CSV file
        target_column (str): Name of target column
        test_size (float): Proportion of test set
        
    Returns:
        dict: Dictionary containing all preprocessed data and scaler
    """
    # Load data
    df = load_data(filepath)
    if df is None:
        return None
    
    # Check and handle missing values
    check_missing_values(df)
    df = handle_missing_values(df, strategy='mean')
    
    # Prepare features and target
    X, y = prepare_features(df, target_column)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print("\n🎉 Preprocessing complete!")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }


if __name__ == "__main__":
    # Example usage
    print("🔍 Testing preprocessing pipeline...\n")
    
    # This will be used once we have the dataset
    # data = preprocess_pipeline('data/heart_disease.csv')
    
    print("\n✅ Preprocessing module loaded successfully!")
    print("📝 Import this module in your training script:")
    print("   from preprocessing import preprocess_pipeline")
