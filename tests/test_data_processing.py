"""
Unit tests for data processing module
"""
import pytest
import numpy as np
from data_processing import (
    load_data, 
    validate_data, 
    preprocess_data, 
    get_data_statistics
)


class TestDataLoading:
    """Test data loading functionality"""
    
    def test_load_data_returns_correct_shapes(self):
        """Test that loaded data has expected shape"""
        X, y = load_data()
        assert X.shape[0] == 150, "Should have 150 samples"
        assert X.shape[1] == 4, "Should have 4 features"
        assert len(y) == 150, "Labels should have 150 samples"
    
    def test_load_data_returns_numpy_arrays(self):
        """Test that data is returned as numpy arrays"""
        X, y = load_data()
        assert isinstance(X, np.ndarray), "X should be numpy array"
        assert isinstance(y, np.ndarray), "y should be numpy array"
    
    def test_load_data_has_three_classes(self):
        """Test that data has exactly 3 classes"""
        X, y = load_data()
        unique_classes = np.unique(y)
        assert len(unique_classes) == 3, "Should have 3 classes"
        assert set(unique_classes) == {0, 1, 2}, "Classes should be 0, 1, 2"


class TestDataValidation:
    """Test data validation functionality"""
    
    def test_validate_data_with_valid_data(self):
        """Test validation passes with valid data"""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([0, 1])
        assert validate_data(X, y) == True
    
    def test_validate_data_with_none_X(self):
        """Test validation fails with None X"""
        with pytest.raises(ValueError, match="Data cannot be None"):
            validate_data(None, np.array([1, 2]))
    
    def test_validate_data_with_none_y(self):
        """Test validation fails with None y"""
        with pytest.raises(ValueError, match="Data cannot be None"):
            validate_data(np.array([[1, 2]]), None)
    
    def test_validate_data_with_empty_X(self):
        """Test validation fails with empty X"""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            validate_data(np.array([]), np.array([1, 2]))
    
    def test_validate_data_with_empty_y(self):
        """Test validation fails with empty y"""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            validate_data(np.array([[1, 2]]), np.array([]))
    
    def test_validate_data_with_mismatched_lengths(self):
        """Test validation fails when X and y have different lengths"""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="Feature and label length mismatch"):
            validate_data(X, y)
    
    def test_validate_data_with_nan_in_X(self):
        """Test validation fails with NaN in features"""
        X = np.array([[1, 2], [np.nan, 4]])
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="Features contain NaN values"):
            validate_data(X, y)
    
    def test_validate_data_with_nan_in_y(self):
        """Test validation fails with NaN in labels"""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, np.nan])
        with pytest.raises(ValueError, match="Labels contain NaN values"):
            validate_data(X, y)


class TestDataPreprocessing:
    """Test data preprocessing functionality"""
    
    def test_preprocess_data_returns_correct_split(self):
        """Test that data is split correctly"""
        X, y = load_data()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, test_size=0.2)
        
        assert len(X_train) == 120, "Training set should have 120 samples"
        assert len(X_test) == 30, "Test set should have 30 samples"
        assert len(y_train) == 120, "Training labels should have 120 samples"
        assert len(y_test) == 30, "Test labels should have 30 samples"
    
    def test_preprocess_data_scales_features(self):
        """Test that features are scaled (mean ~0, std ~1)"""
        X, y = load_data()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        
        # Check that training data is approximately standardized
        train_mean = np.mean(X_train, axis=0)
        train_std = np.std(X_train, axis=0)
        
        assert np.allclose(train_mean, 0, atol=0.1), "Mean should be close to 0"
        assert np.allclose(train_std, 1, atol=0.1), "Std should be close to 1"
    
    def test_preprocess_data_with_custom_test_size(self):
        """Test preprocessing with custom test size"""
        X, y = load_data()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, test_size=0.3)
        
        assert len(X_train) == 105, "Training set should have 105 samples"
        assert len(X_test) == 45, "Test set should have 45 samples"


class TestDataStatistics:
    """Test data statistics functionality"""
    
    def test_get_data_statistics_returns_correct_info(self):
        """Test that statistics are computed correctly"""
        X, y = load_data()
        stats = get_data_statistics(X, y)
        
        assert stats['n_samples'] == 150, "Should report 150 samples"
        assert stats['n_features'] == 4, "Should report 4 features"
        assert stats['n_classes'] == 3, "Should report 3 classes"
    
    def test_get_data_statistics_class_distribution(self):
        """Test that class distribution is balanced"""
        X, y = load_data()
        stats = get_data_statistics(X, y)
        
        distribution = stats['class_distribution']
        assert distribution[0] == 50, "Class 0 should have 50 samples"
        assert distribution[1] == 50, "Class 1 should have 50 samples"
        assert distribution[2] == 50, "Class 2 should have 50 samples"


# Test to verify all tests can be discovered
def test_suite_completeness():
    """Verify test suite is complete"""
    assert True, "Test suite loaded successfully"
