"""
Prediction Functions for Heart Disease Predictor

This module contains functions for loading trained models and making predictions.
"""

import joblib
import numpy as np
import os


def load_model(model_path='models/heart_model.pkl'):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        object: Loaded model, or None if loading fails
    """
    try:
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at {model_path}")
            print("💡 Tip: Train a model first using train_model.py")
            return None
        
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None


def load_scaler(scaler_path='models/scaler.pkl'):
    """
    Load the feature scaler from disk.
    
    Args:
        scaler_path (str): Path to the saved scaler file
        
    Returns:
        object: Loaded scaler, or None if loading fails
    """
    try:
        if not os.path.exists(scaler_path):
            print(f"⚠️ Scaler file not found at {scaler_path}")
            return None
        
        scaler = joblib.load(scaler_path)
        print(f"✅ Scaler loaded successfully from {scaler_path}")
        return scaler
    except Exception as e:
        print(f"❌ Error loading scaler: {str(e)}")
        return None


def predict_single(model, scaler, features):
    """
    Make a prediction for a single patient.
    
    Args:
        model: Trained model
        scaler: Feature scaler (optional, can be None)
        features (array-like): Patient features
        
    Returns:
        tuple: (prediction, probability) - Predicted class and probability
    """
    try:
        # Convert to numpy array if not already
        features = np.array(features).reshape(1, -1)
        
        # Scale features if scaler is provided
        if scaler is not None:
            features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features)[0]
            confidence = max(probability) * 100
        else:
            probability = None
            confidence = None
        
        return prediction, probability, confidence
    
    except Exception as e:
        print(f"❌ Error making prediction: {str(e)}")
        return None, None, None


def predict_batch(model, scaler, features_list):
    """
    Make predictions for multiple patients.
    
    Args:
        model: Trained model
        scaler: Feature scaler (optional, can be None)
        features_list (array-like): List of patient features
        
    Returns:
        np.array: Array of predictions
    """
    try:
        # Convert to numpy array
        features_array = np.array(features_list)
        
        # Scale features if scaler is provided
        if scaler is not None:
            features_array = scaler.transform(features_array)
        
        # Make predictions
        predictions = model.predict(features_array)
        
        print(f"✅ Made predictions for {len(predictions)} patients")
        return predictions
    
    except Exception as e:
        print(f"❌ Error making batch predictions: {str(e)}")
        return None


def interpret_prediction(prediction, probability=None):
    """
    Interpret the model's prediction into human-readable form.
    
    Args:
        prediction (int): Model prediction (0 or 1)
        probability (array, optional): Probability distribution
        
    Returns:
        dict: Interpretation with risk level and recommendation
    """
    if prediction == 1:
        risk_level = "High Risk"
        message = "The model indicates a higher likelihood of heart disease."
        recommendation = "⚠️ Please consult with a healthcare professional for proper diagnosis and treatment."
        color = "red"
    else:
        risk_level = "Low Risk"
        message = "The model indicates a lower likelihood of heart disease."
        recommendation = "✅ Continue maintaining a healthy lifestyle. Regular check-ups are still recommended."
        color = "green"
    
    result = {
        'risk_level': risk_level,
        'message': message,
        'recommendation': recommendation,
        'color': color,
        'prediction': int(prediction)
    }
    
    if probability is not None:
        result['probability_no_disease'] = float(probability[0])
        result['probability_disease'] = float(probability[1])
        result['confidence'] = float(max(probability)) * 100
    
    return result


if __name__ == "__main__":
    # Example usage
    print("🔍 Testing prediction module...\n")
    
    # Try to load model (will fail if not trained yet)
    model = load_model()
    scaler = load_scaler()
    
    if model is None:
        print("\n💡 Train a model first by running: python train_model.py")
    else:
        print("\n✅ Prediction module ready!")
        print("📝 Example usage:")
        print("   from predict import load_model, predict_single")
        print("   model = load_model()")
        print("   prediction, probability, confidence = predict_single(model, scaler, features)")
