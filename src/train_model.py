"""
Model training module for ML pipeline
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_processing import load_data, preprocess_data, get_data_statistics


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees in the forest
        random_state: Random seed for reproducibility
        
    Returns:
        Trained model
    """
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=5,
        min_samples_split=5
    )
    model.fit(X_train, y_train)
    print("Training complete!")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred
    }


def save_model(model, scaler, filepath='model.pkl'):
    """
    Save the trained model and scaler.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        filepath: Path to save the model
    """
    model_data = {
        'model': model,
        'scaler': scaler
    }
    joblib.dump(model_data, filepath)
    print(f"\nModel saved to {filepath}")


def main():
    """
    Main training pipeline.
    """
    print("=" * 50)
    print("ML Training Pipeline")
    print("=" * 50)
    
    # Load data
    print("\n1. Loading data...")
    X, y = load_data()
    stats = get_data_statistics(X, y)
    print(f"   Dataset: {stats['n_samples']} samples, {stats['n_features']} features")
    print(f"   Classes: {stats['n_classes']}")
    print(f"   Distribution: {stats['class_distribution']}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Train model
    print("\n3. Training model...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("\n4. Evaluating model...")
    results = evaluate_model(model, X_test, y_test)
    
    # Save model
    print("\n5. Saving model...")
    save_model(model, scaler, 'model.pkl')
    
    # Minimum accuracy requirement
    if results['accuracy'] < 0.7:
        print("\n⚠️  WARNING: Model accuracy below threshold (0.7)")
        exit(1)
    else:
        print(f"\n✓ Model meets accuracy requirement: {results['accuracy']:.4f} >= 0.7")
    
    print("\n" + "=" * 50)
    print("Pipeline complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
