"""
Custom Exception Classes
Provides specific exception types for better error handling and debugging
"""

from typing import Optional, Dict, Any


class CreateLLMError(Exception):
    """
    Base exception for all create-llm errors
    
    Provides consistent error formatting with helpful suggestions
    """
    
    def __init__(
        self,
        message: str,
        suggestion: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize error
        
        Args:
            message: Error message
            suggestion: Optional suggestion for fixing the error
            details: Optional additional details
        """
        self.message = message
        self.suggestion = suggestion
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Format error message with suggestion"""
        error_msg = f"❌ {self.__class__.__name__}: {self.message}"
        
        if self.suggestion:
            error_msg += f"\n\n💡 Suggestion: {self.suggestion}"
        
        if self.details:
            error_msg += "\n\n📋 Details:"
            for key, value in self.details.items():
                error_msg += f"\n   • {key}: {value}"
        
        return error_msg


class ConfigurationError(CreateLLMError):
    """
    Raised when there's an error in configuration
    
    Examples:
        - Invalid config file
        - Missing required fields
        - Invalid parameter values
        - Hardware incompatibility
    """
    pass


class DataError(CreateLLMError):
    """
    Raised when there's an error with data
    
    Examples:
        - Missing data files
        - Invalid data format
        - Tokenization failures
        - Empty dataset
    """
    pass


class TrainingError(CreateLLMError):
    """
    Raised when there's an error during training
    
    Examples:
        - NaN/Inf losses
        - Out of memory
        - Checkpoint corruption
        - Model convergence issues
    """
    pass


class ModelError(CreateLLMError):
    """
    Raised when there's an error with the model
    
    Examples:
        - Model loading failures
        - Architecture mismatches
        - Invalid model state
        - Forward pass errors
    """
    pass


class DeploymentError(CreateLLMError):
    """
    Raised when there's an error during deployment
    
    Examples:
        - Authentication failures
        - Upload errors
        - Network issues
        - Invalid credentials
    """
    pass


class PluginError(CreateLLMError):
    """
    Raised when there's an error with plugins
    
    Examples:
        - Plugin loading failures
        - Plugin initialization errors
        - Plugin execution errors
    """
    pass


# Error message templates
ERROR_MESSAGES = {
    'config_not_found': {
        'message': 'Configuration file not found: {path}',
        'suggestion': 'Make sure llm.config.js exists in the project root directory'
    },
    'config_invalid': {
        'message': 'Invalid configuration: {reason}',
        'suggestion': 'Check your llm.config.js for syntax errors or missing fields'
    },
    'data_not_found': {
        'message': 'Data file not found: {path}',
        'suggestion': 'Place your training data in data/raw/ directory'
    },
    'data_empty': {
        'message': 'Dataset is empty or too small',
        'suggestion': 'Provide at least 1MB of text data for training'
    },
    'tokenizer_not_found': {
        'message': 'Tokenizer not found: {path}',
        'suggestion': 'Train a tokenizer first: python tokenizer/train.py --data data/raw/sample.txt'
    },
    'checkpoint_not_found': {
        'message': 'Checkpoint not found: {path}',
        'suggestion': 'Check the checkpoint path or train a model first'
    },
    'checkpoint_corrupted': {
        'message': 'Checkpoint file is corrupted: {path}',
        'suggestion': 'Try loading a different checkpoint or retrain the model'
    },
    'out_of_memory': {
        'message': 'Out of memory during {operation}',
        'suggestion': 'Try reducing batch_size, max_length, or use gradient_accumulation in llm.config.js'
    },
    'nan_loss': {
        'message': 'Loss became NaN at step {step}',
        'suggestion': 'Try reducing learning_rate, enabling gradient_clip, or checking your data for issues'
    },
    'gpu_not_available': {
        'message': 'GPU not available, falling back to CPU',
        'suggestion': 'Training will be slower on CPU. Consider using a GPU or reducing model size'
    },
    'plugin_failed': {
        'message': 'Plugin {plugin_name} failed to load: {reason}',
        'suggestion': 'Training will continue without this plugin. Check plugin configuration or installation'
    },
    'deployment_auth_failed': {
        'message': 'Authentication failed for {platform}',
        'suggestion': 'Login with: {login_command}'
    },
}


def get_error_message(error_type: str, **kwargs) -> tuple:
    """
    Get formatted error message and suggestion
    
    Args:
        error_type: Type of error from ERROR_MESSAGES
        **kwargs: Format arguments for the message
    
    Returns:
        Tuple of (message, suggestion)
    """
    if error_type not in ERROR_MESSAGES:
        return (f"Unknown error: {error_type}", None)
    
    template = ERROR_MESSAGES[error_type]
    message = template['message'].format(**kwargs)
    suggestion = template['suggestion'].format(**kwargs) if 'suggestion' in template else None
    
    return (message, suggestion)
