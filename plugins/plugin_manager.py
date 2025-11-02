"""
Plugin Manager
Manages loading, initialization, and lifecycle of plugins
"""

import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import Plugin, PluginError


class PluginManager:
    """
    Manages plugins for the LLM training system
    
    Loads plugins from config, initializes them, and provides
    methods to call plugin lifecycle hooks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize plugin manager
        
        Args:
            config: Configuration dictionary from llm.config.js
        """
        self.config = config
        self.plugins: Dict[str, Plugin] = {}
        self.failed_plugins: List[str] = []
        
        # Load plugins
        self._load_plugins()
    
    def _load_plugins(self) -> None:
        """Load plugins specified in config"""
        plugin_names = self.config.get('plugins', [])
        
        if not plugin_names:
            print("ℹ️  No plugins configured")
            return
        
        print(f"\n📦 Loading {len(plugin_names)} plugin(s)...")
        
        for plugin_name in plugin_names:
            try:
                self._load_plugin(plugin_name)
            except Exception as e:
                self.failed_plugins.append(plugin_name)
                print(f"⚠️  Warning: Failed to load plugin '{plugin_name}': {e}")
                print(f"   Training will continue without this plugin")
        
        # Summary
        loaded_count = len(self.plugins)
        failed_count = len(self.failed_plugins)
        
        if loaded_count > 0:
            print(f"✓ Successfully loaded {loaded_count} plugin(s)")
            for name in self.plugins.keys():
                print(f"  • {name}")
        
        if failed_count > 0:
            print(f"⚠️  Failed to load {failed_count} plugin(s): {', '.join(self.failed_plugins)}")
    
    def _load_plugin(self, plugin_name: str) -> None:
        """
        Load a single plugin
        
        Args:
            plugin_name: Name of the plugin to load
        
        Raises:
            PluginError: If plugin cannot be loaded
        """
        # Try to import plugin from plugins directory
        try:
            # Add plugins directory to path if not already there
            plugins_dir = Path('plugins')
            if plugins_dir.exists() and str(plugins_dir) not in sys.path:
                sys.path.insert(0, str(plugins_dir))
            
            # Import plugin module
            module_name = f"{plugin_name}_plugin"
            module = importlib.import_module(module_name)
            
            # Get plugin class (should be named like WandBPlugin, SynthexPlugin, etc.)
            class_name = ''.join(word.capitalize() for word in plugin_name.split('_')) + 'Plugin'
            
            if not hasattr(module, class_name):
                raise PluginError(f"Plugin module '{module_name}' does not have class '{class_name}'")
            
            plugin_class = getattr(module, class_name)
            
            # Instantiate plugin
            plugin = plugin_class(name=plugin_name)
            
            # Initialize plugin
            if not plugin.initialize(self.config):
                raise PluginError(f"Plugin '{plugin_name}' initialization failed")
            
            # Store plugin
            self.plugins[plugin_name] = plugin
            
        except ImportError as e:
            raise PluginError(f"Could not import plugin '{plugin_name}': {e}")
        except Exception as e:
            raise PluginError(f"Error loading plugin '{plugin_name}': {e}")
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Get a loaded plugin by name
        
        Args:
            name: Plugin name
        
        Returns:
            Plugin instance or None if not found
        """
        return self.plugins.get(name)
    
    def has_plugin(self, name: str) -> bool:
        """
        Check if a plugin is loaded
        
        Args:
            name: Plugin name
        
        Returns:
            True if plugin is loaded, False otherwise
        """
        return name in self.plugins
    
    def get_all_plugins(self) -> List[Plugin]:
        """
        Get all loaded plugins
        
        Returns:
            List of plugin instances
        """
        return list(self.plugins.values())
    
    # Lifecycle hook methods
    
    def on_train_begin(self, trainer) -> None:
        """Call on_train_begin for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_train_begin(trainer)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_train_begin: {e}")
    
    def on_train_end(self, trainer, final_metrics: Dict[str, Any]) -> None:
        """Call on_train_end for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_train_end(trainer, final_metrics)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_train_end: {e}")
    
    def on_epoch_begin(self, trainer, epoch: int) -> None:
        """Call on_epoch_begin for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_epoch_begin(trainer, epoch)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_epoch_begin: {e}")
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]) -> None:
        """Call on_epoch_end for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_epoch_end(trainer, epoch, metrics)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_epoch_end: {e}")
    
    def on_step_begin(self, trainer, step: int) -> None:
        """Call on_step_begin for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_step_begin(trainer, step)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_step_begin: {e}")
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, Any]) -> None:
        """Call on_step_end for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_step_end(trainer, step, metrics)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_step_end: {e}")
    
    def on_validation_begin(self, trainer) -> None:
        """Call on_validation_begin for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_validation_begin(trainer)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_validation_begin: {e}")
    
    def on_validation_end(self, trainer, metrics: Dict[str, Any]) -> None:
        """Call on_validation_end for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_validation_end(trainer, metrics)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_validation_end: {e}")
    
    def on_checkpoint_save(self, trainer, checkpoint_path: str) -> None:
        """Call on_checkpoint_save for all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.on_checkpoint_save(trainer, checkpoint_path)
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in on_checkpoint_save: {e}")
    
    def cleanup(self) -> None:
        """Cleanup all plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                print(f"⚠️  Plugin '{plugin.name}' error in cleanup: {e}")
    
    def __repr__(self) -> str:
        return f"<PluginManager: {len(self.plugins)} plugins loaded>"


def create_plugin_manager(config: Dict[str, Any]) -> PluginManager:
    """
    Create and initialize plugin manager
    
    Args:
        config: Configuration dictionary
    
    Returns:
        PluginManager instance
    """
    return PluginManager(config)
