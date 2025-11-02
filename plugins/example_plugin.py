"""
Example Plugin
Template for creating custom plugins
"""

from typing import Dict, Any
from .base import Plugin


class ExamplePlugin(Plugin):
    """
    Example plugin demonstrating the plugin interface
    
    To create your own plugin:
    1. Copy this file to plugins/your_plugin_name_plugin.py
    2. Rename the class to YourPluginNamePlugin
    3. Implement the initialize() method and any lifecycle hooks you need
    4. Add 'your_plugin_name' to the plugins list in llm.config.js
    """
    
    def __init__(self, name: str = 'example'):
        super().__init__(name)
        self.step_count = 0
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize plugin with configuration
        
        Args:
            config: Full configuration from llm.config.js
        
        Returns:
            True if successful, False otherwise
        """
        print(f"Initializing {self.name} plugin...")
        
        # Store any plugin-specific config
        self.config = config
        
        # Perform any setup needed
        # Return False if initialization fails
        
        print(f"✓ {self.name} plugin initialized")
        return True
    
    def on_train_begin(self, trainer) -> None:
        """Called when training begins"""
        print(f"[{self.name}] Training started")
        self.step_count = 0
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, Any]) -> None:
        """Called after each training step"""
        self.step_count += 1
        
        # Example: Log every 100 steps
        if step % 100 == 0:
            print(f"[{self.name}] Step {step}: loss={metrics.get('loss', 0):.4f}")
    
    def on_train_end(self, trainer, final_metrics: Dict[str, Any]) -> None:
        """Called when training ends"""
        print(f"[{self.name}] Training completed after {self.step_count} steps")
        print(f"[{self.name}] Final loss: {final_metrics.get('loss', 0):.4f}")
    
    def cleanup(self) -> None:
        """Cleanup plugin resources"""
        print(f"[{self.name}] Cleaning up...")
