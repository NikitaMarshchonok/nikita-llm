"""
Error Handling Utilities
Provides utilities for graceful error handling and recovery
"""

import functools
import time
import torch
from typing import Callable, Optional, Any, Type
from pathlib import Path


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry a function on failure
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
    
    Example:
        @retry_on_failure(max_retries=3, delay=1.0)
        def save_checkpoint(path):
            torch.save(model.state_dict(), path)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        print(f"⚠️  Attempt {attempt + 1}/{max_retries} failed: {e}")
                        print(f"   Retrying in {current_delay:.1f}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        print(f"❌ All {max_retries} attempts failed")
            
            raise last_exception
        
        return wrapper
    return decorator


def safe_gpu_operation(fallback_to_cpu: bool = True):
    """
    Decorator to safely handle GPU operations with CPU fallback
    
    Args:
        fallback_to_cpu: Whether to fallback to CPU on GPU errors
    
    Example:
        @safe_gpu_operation(fallback_to_cpu=True)
        def train_step(model, batch):
            return model(batch)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"\n⚠️  GPU out of memory!")
                    print(f"💡 Suggestion: Reduce batch_size or max_length in llm.config.js")
                    
                    if fallback_to_cpu and torch.cuda.is_available():
                        print(f"   Attempting to clear GPU cache...")
                        torch.cuda.empty_cache()
                        print(f"   Retrying operation...")
                        return func(*args, **kwargs)
                    
                    raise
                else:
                    raise
        
        return wrapper
    return decorator


def validate_path(
    path: str,
    must_exist: bool = True,
    create_if_missing: bool = False,
    path_type: str = 'file'
) -> Path:
    """
    Validate and optionally create a path
    
    Args:
        path: Path to validate
        must_exist: Whether path must exist
        create_if_missing: Create path if it doesn't exist
        path_type: Type of path ('file' or 'directory')
    
    Returns:
        Validated Path object
    
    Raises:
        FileNotFoundError: If path doesn't exist and must_exist=True
    """
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        if create_if_missing:
            if path_type == 'directory':
                path_obj.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created directory: {path}")
            else:
                path_obj.parent.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(
                f"{path_type.capitalize()} not found: {path}\n"
                f"💡 Make sure the path exists and is accessible"
            )
    
    return path_obj


def check_gpu_availability(required: bool = False) -> tuple:
    """
    Check GPU availability and provide helpful messages
    
    Args:
        required: Whether GPU is required
    
    Returns:
        Tuple of (device, gpu_available)
    
    Raises:
        RuntimeError: If GPU is required but not available
    """
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✓ GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = 'cpu'
        if required:
            raise RuntimeError(
                "GPU is required for this operation but not available\n"
                "💡 Suggestion: Use a smaller model template or enable CPU training"
            )
        else:
            print(f"⚠️  GPU not available, using CPU")
            print(f"💡 Training will be slower on CPU")
    
    return device, gpu_available


def handle_nan_loss(loss: torch.Tensor, step: int) -> None:
    """
    Check for NaN loss and provide helpful error message
    
    Args:
        loss: Loss tensor
        step: Current training step
    
    Raises:
        TrainingError: If loss is NaN or Inf
    """
    from .exceptions import TrainingError
    
    if torch.isnan(loss):
        raise TrainingError(
            f"Loss became NaN at step {step}",
            suggestion=(
                "Try these fixes:\n"
                "   1. Reduce learning_rate in llm.config.js\n"
                "   2. Enable gradient_clip (e.g., gradient_clip: 1.0)\n"
                "   3. Check your data for invalid values\n"
                "   4. Reduce batch_size or max_length"
            ),
            details={'step': step, 'loss': 'NaN'}
        )
    
    if torch.isinf(loss):
        raise TrainingError(
            f"Loss became Inf at step {step}",
            suggestion=(
                "Try these fixes:\n"
                "   1. Reduce learning_rate significantly\n"
                "   2. Enable gradient_clip with a lower value\n"
                "   3. Check for numerical instability in your model"
            ),
            details={'step': step, 'loss': 'Inf'}
        )


def safe_checkpoint_save(
    state_dict: dict,
    path: str,
    max_retries: int = 3
) -> bool:
    """
    Safely save checkpoint with retry logic
    
    Args:
        state_dict: State dictionary to save
        path: Path to save checkpoint
        max_retries: Maximum retry attempts
    
    Returns:
        True if successful, False otherwise
    """
    @retry_on_failure(max_retries=max_retries, delay=1.0)
    def _save():
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to temporary file first
        temp_path = f"{path}.tmp"
        torch.save(state_dict, temp_path)
        
        # Verify the save
        try:
            torch.load(temp_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Checkpoint verification failed: {e}")
        
        # Move to final location
        Path(temp_path).replace(path)
    
    try:
        _save()
        return True
    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")
        return False


def graceful_shutdown(cleanup_func: Optional[Callable] = None):
    """
    Decorator for graceful shutdown on interruption
    
    Args:
        cleanup_func: Optional cleanup function to call
    
    Example:
        @graceful_shutdown(cleanup_func=save_checkpoint)
        def train():
            # Training code
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                print(f"\n\n⚠️  Training interrupted by user")
                if cleanup_func:
                    print(f"💾 Saving checkpoint before exit...")
                    try:
                        cleanup_func()
                        print(f"✓ Checkpoint saved successfully")
                    except Exception as e:
                        print(f"❌ Failed to save checkpoint: {e}")
                print(f"\n👋 Goodbye!\n")
                raise
        
        return wrapper
    return decorator
