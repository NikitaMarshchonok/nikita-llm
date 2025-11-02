"""
Plugins package
Extensible plugin system for create-llm
"""

from .base import Plugin, PluginError
from .plugin_manager import PluginManager, create_plugin_manager

__all__ = [
    'Plugin',
    'PluginError',
    'PluginManager',
    'create_plugin_manager',
]
