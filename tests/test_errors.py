"""
Tests for error handling
"""

import pytest
from utils.exceptions import (
    CreateLLMError,
    ConfigurationError,
    DataError,
    TrainingError,
)
from utils.handlers import retry_on_failure, validate_path


class TestExceptions:
    """Tests for custom exceptions"""
    
    def test_base_exception(self):
        """Test base exception"""
        error = CreateLLMError("Test error", suggestion="Try this")
        assert "Test error" in str(error)
        assert "Try this" in str(error)
    
    def test_configuration_error(self):
        """Test configuration error"""
        error = ConfigurationError("Invalid config")
        assert isinstance(error, CreateLLMError)
    
    def test_data_error(self):
        """Test data error"""
        error = DataError("Data not found")
        assert isinstance(error, CreateLLMError)
    
    def test_training_error(self):
        """Test training error"""
        error = TrainingError("Training failed")
        assert isinstance(error, CreateLLMError)


class TestErrorHandlers:
    """Tests for error handling utilities"""
    
    def test_retry_on_failure(self):
        """Test retry decorator"""
        call_count = 0
        
        @retry_on_failure(max_retries=3, delay=0.1)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Test error")
            return "success"
        
        result = failing_function()
        assert result == "success"
        assert call_count == 3
    
    def test_validate_path_exists(self, temp_dir):
        """Test path validation for existing path"""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        
        validated = validate_path(str(test_file), must_exist=True)
        assert validated.exists()
    
    def test_validate_path_not_exists(self, temp_dir):
        """Test path validation for non-existing path"""
        test_file = temp_dir / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError):
            validate_path(str(test_file), must_exist=True)
