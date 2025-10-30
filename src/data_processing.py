"""
Data processing module for ML pipeline
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data():
    """
    Load sample data for demonstration.
    In a real project, this would load from a file or database.
    """
    # Generate synthetic data for iris-like classification
    np.random.seed(42)
    
    # Create 3 classes with different characteristics
    n_samples_per_class = 50
    
    # Class 0: small values
    class_0 = np.random.randn(n_samples_per_class, 4) * 0.5 + [5.0, 3.0, 1.5, 0.2]
    
    # Class 1: medium values
    class_1 = np.random.randn(n_samples_per_class, 4) * 0.5 + [6.0, 2.8, 4.5, 1.3]
    
    # Class 2: large values
    class_2 = np.random.randn(n_samples_per_class, 4) * 0.5 + [6.5, 3.0, 5.5, 2.0]
    
    # Combine data
    X = np.vstack([class_0, class_1, class_2])
    y = np.array([0] * n_samples_per_class + 
                  [1] * n_samples_per_class + 
                  [2] * n_samples_per_class)
    
    return X, y


def validate_data(X, y):
    """
    Validate that data meets requirements.
    
    Args:
        X: Feature matrix
        y: Target labels
        
    Returns:
        bool: True if data is valid
        
    Raises:
        ValueError: If data validation fails
    """
    if X is None or y is None:
        raise ValueError("Data cannot be None")
    
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Data cannot be empty")
    
    if len(X) != len(y):
        raise ValueError(f"Feature and label length mismatch: {len(X)} != {len(y)}")
    
    if np.any(np.isnan(X)):
        raise ValueError("Features contain NaN values")
    
    if np.any(np.isnan(y)):
        raise ValueError("Labels contain NaN values")
    
    return True


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Preprocess data: validate, split, and scale.
    
    Args:
        X: Feature matrix
        y: Target labels
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    # Validate data
    validate_data(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_data_statistics(X, y):
    """
    Get basic statistics about the dataset.
    
    Args:
        X: Feature matrix
        y: Target labels
        
    Returns:
        dict: Dictionary containing data statistics
    """
    stats = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'class_distribution': {int(cls): int(np.sum(y == cls)) for cls in np.unique(y)}
    }
    return stats
