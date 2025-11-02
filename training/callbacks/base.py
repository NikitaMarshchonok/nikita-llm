"""
Base callback class for training lifecycle hooks
"""

from typing import Any, Dict, Optional


class Callback:
    """
    Base class for training callbacks
    
    Callbacks allow you to hook into the training loop at various points:
    - on_train_begin: Called at the start of training
    - on_train_end: Called at the end of training
    - on_epoch_begin: Called at the start of each epoch
    - on_epoch_end: Called at the end of each epoch
    - on_batch_begin: Called before processing each batch
    - on_batch_end: Called after processing each batch
    - on_step_end: Called after each training step
    """
    
    def on_train_begin(self, trainer: Any):
        """Called when training begins"""
        pass
    
    def on_train_end(self, trainer: Any):
        """Called when training ends"""
        pass
    
    def on_epoch_begin(self, trainer: Any, epoch: int):
        """Called at the beginning of each epoch"""
        pass
    
    def on_epoch_end(self, trainer: Any, epoch: int):
        """Called at the end of each epoch"""
        pass
    
    def on_batch_begin(self, trainer: Any, batch: Dict[str, Any]):
        """Called before processing each batch"""
        pass
    
    def on_batch_end(self, trainer: Any, batch: Dict[str, Any], outputs: Dict[str, Any]):
        """Called after processing each batch"""
        pass
    
    def on_step_end(self, trainer: Any, step: int, loss: float, metrics: Optional[Dict[str, float]] = None):
        """Called after each training step"""
        pass
