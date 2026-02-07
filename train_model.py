"""
Model Training Script for Heart Disease Predictor

This script trains multiple machine learning models on the heart disease dataset
and saves the best performing model.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from preprocessing import preprocess_pipeline


def train_models(X_train, y_train):
    """
    Train multiple machine learning models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        dict: Dictionary of trained models
    """
    print("\n🤖 Training multiple models...\n")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✅ {name} trained")
    
    print("\n🎉 All models trained successfully!\n")
    return trained_models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all trained models and return performance metrics.
    
    Args:
        models (dict): Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        
    Returns:
        pd.DataFrame: Performance metrics for all models
    """
    print("📊 Evaluating models...\n")
    
    results = []
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })
        
        print(f"✅ {name}:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        if roc_auc:
            print(f"   ROC-AUC: {roc_auc:.4f}")
        print()
    
    results_df = pd.DataFrame(results)
    return results_df


def get_best_model(results_df, models, metric='Accuracy'):
    """
    Get the best performing model based on specified metric.
    
    Args:
        results_df (pd.DataFrame): Model evaluation results
        models (dict): Dictionary of trained models
        metric (str): Metric to use for selection
        
    Returns:
        tuple: (best_model_name, best_model, best_score)
    """
    best_idx = results_df[metric].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_score = results_df.loc[best_idx, metric]
    best_model = models[best_model_name]
    
    print(f"🏆 Best Model: {best_model_name}")
    print(f"   {metric}: {best_score:.4f}\n")
    
    return best_model_name, best_model, best_score


def save_model(model, scaler, model_path='models/heart_model.pkl', scaler_path='models/scaler.pkl'):
    """
    Save trained model and scaler to disk.
    
    Args:
        model: Trained model
        scaler: Feature scaler
        model_path (str): Path to save model
        scaler_path (str): Path to save scaler
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Save scaler
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to {scaler_path}")


def main():
    """
    Main training pipeline.
    """
    print("=" * 60)
    print("🫀 HEART DISEASE PREDICTOR - MODEL TRAINING")
    print("=" * 60)
    print(f"\n⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check if data file exists
    data_path = 'data/heart_disease.csv'
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found at {data_path}")
        print("\n💡 Please download the heart disease dataset and place it in the data/ folder")
        print("   You can get it from: https://archive.ics.uci.edu/ml/datasets/heart+disease")
        return
    
    # Preprocess data
    print("📥 Loading and preprocessing data...\n")
    data = preprocess_pipeline(data_path, target_column='target', test_size=0.2)
    
    if data is None:
        print("❌ Preprocessing failed. Exiting.")
        return
    
    # Extract preprocessed data
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    scaler = data['scaler']
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Display results table
    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON")
    print("=" * 60)
    print(results.to_string(index=False))
    print()
    
    # Select best model
    best_name, best_model, best_score = get_best_model(results, models, metric='Accuracy')
    
    # Save best model
    save_model(best_model, scaler)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n💾 Saved: {best_name} (Accuracy: {best_score:.4f})")
    print(f"📍 Location: models/heart_model.pkl")
    print(f"\n🚀 Next step: Run the Streamlit app")
    print(f"   Command: streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
