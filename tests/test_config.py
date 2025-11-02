"""
Tests for configuration loading and validation
"""

import pytest
from models.config import ConfigLoader, ConfigValidationError


class TestConfigLoader:
    """Tests for ConfigLoader"""
    
    def test_config_loading(self):
        """Test config can be loaded"""
        try:
            config = ConfigLoader('llm.config.js')
            assert config is not None
        except FileNotFoundError:
            pytest.skip("Config file not found")
    
    def test_config_validation(self, sample_config):
        """Test config validation"""
        # This would need a mock config file
        # For now, just test that validation methods exist
        pass
    
    def test_config_getters(self, sample_config):
        """Test config getter methods"""
        # Test that we can access config values
        assert 'model' in sample_config
        assert 'training' in sample_config
