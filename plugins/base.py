"""
Plugin Base Class
Base class for all create-llm plugins
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class Plugin(ABC):
    """
    Base class for create-llm plugins
    
    Plugins can hook into various stages of the training lifecycle
    to add custom functionality like logging, monitoring, or data generation.
    """
    
    def __init__(self, name: str):
        """
        Initialize plugin
        
        Args:
            name: Plugin name
        """
        self.name = name
        self.enabled = True
        self.config: Dict[str, Any] = {}
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize plugin with configuration
        
        Args:
            config: Full configuration dictionary from llm.config.js
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    def on_train_begin(self, trainer) -> None:
        """
        Called when training begins
        
        Args:
            trainer: Trainer instance
        """
        pass
    
    def on_train_end(self, trainer, final_metrics: Dict[str, Any]) -> None:
        """
        Called when training ends
        
        Args:
            trainer: Trainer instance
            final_metrics: Final training metrics
        """
        pass
    
    def on_epoch_begin(self, trainer, epoch: int) -> None:
        """
        Called at the beginning of each epoch
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch number
        """
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Called at the end of each epoch
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch number
            metrics: Epoch metrics
        """
        pass
    
    def on_step_begin(self, trainer, step: int) -> None:
        """
        Called at the beginning of each training step
        
        Args:
            trainer: Trainer instance
            step: Current step number
        """
        pass
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, Any]) -> None:
        """
        Called at the end of each training step
        
        Args:
            trainer: Trainer instance
            step: Current step number
            metrics: Step metrics (loss, lr, etc.)
        """
        pass
    
    def on_validation_begin(self, trainer) -> None:
        """
        Called when validation begins
        
        Args:
            trainer: Trainer instance
        """
        pass
    
    def on_validation_end(self, trainer, metrics: Dict[str, Any]) -> None:
        """
        Called when validation ends
        
        Args:
            trainer: Trainer instance
            metrics: Validation metrics
        """
        pass
    
    def on_checkpoint_save(self, trainer, checkpoint_path: str) -> None:
        """
        Called when a checkpoint is saved
        
        Args:
            trainer: Trainer instance
            checkpoint_path: Path to saved checkpoint
        """
        pass
    
    def cleanup(self) -> None:
        """
        Cleanup plugin resources
        Called when plugin is being unloaded
        """
        pass
    
    def __repr__(self) -> str:
        return f"<Plugin: {self.name} (enabled={self.enabled})>"


class PluginError(Exception):
    """Exception raised for plugin-related errors"""
    pass
