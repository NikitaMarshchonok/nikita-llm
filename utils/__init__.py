"""
Error handling package
Provides custom exceptions and error handling utilities
"""

from .exceptions import (
    CreateLLMError,
    ConfigurationError,
    DataError,
    TrainingError,
    ModelError,
    DeploymentError,
    PluginError,
    get_error_message,
    ERROR_MESSAGES,
)

from .handlers import (
    retry_on_failure,
    safe_gpu_operation,
    validate_path,
    check_gpu_availability,
    handle_nan_loss,
    safe_checkpoint_save,
    graceful_shutdown,
)

__all__ = [
    # Exceptions
    'CreateLLMError',
    'ConfigurationError',
    'DataError',
    'TrainingError',
    'ModelError',
    'DeploymentError',
    'PluginError',
    'get_error_message',
    'ERROR_MESSAGES',
    # Handlers
    'retry_on_failure',
    'safe_gpu_operation',
    'validate_path',
    'check_gpu_availability',
    'handle_nan_loss',
    'safe_checkpoint_save',
    'graceful_shutdown',
]
