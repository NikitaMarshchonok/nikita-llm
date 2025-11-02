# Plugins

This directory contains plugins for extending create-llm functionality.

## Available Plugins

### Built-in Plugins

- **wandb**: Weights & Biases integration for experiment tracking
- **synthex**: SynthexAI integration for synthetic data generation
- **huggingface**: Hugging Face Hub integration for model sharing

### Custom Plugins

You can create custom plugins by following the example in `example_plugin.py`.

## Creating a Custom Plugin

1. Create a new file: `plugins/my_plugin_plugin.py`
2. Define a class that inherits from `Plugin`:

```python
from plugins.base import Plugin

class MyPluginPlugin(Plugin):
    def __init__(self, name='my_plugin'):
        super().__init__(name)
    
    def initialize(self, config):
        # Setup your plugin
        return True
    
    def on_step_end(self, trainer, step, metrics):
        # Do something on each training step
        pass
```

3. Add your plugin to `llm.config.js`:

```javascript
module.exports = {
  // ... other config
  plugins: [
    'my_plugin',
  ],
};
```

## Plugin Lifecycle Hooks

Plugins can implement the following hooks:

- `initialize(config)`: Called when plugin is loaded
- `on_train_begin(trainer)`: Called when training starts
- `on_train_end(trainer, final_metrics)`: Called when training ends
- `on_epoch_begin(trainer, epoch)`: Called at start of each epoch
- `on_epoch_end(trainer, epoch, metrics)`: Called at end of each epoch
- `on_step_begin(trainer, step)`: Called at start of each step
- `on_step_end(trainer, step, metrics)`: Called at end of each step
- `on_validation_begin(trainer)`: Called when validation starts
- `on_validation_end(trainer, metrics)`: Called when validation ends
- `on_checkpoint_save(trainer, checkpoint_path)`: Called when checkpoint is saved
- `cleanup()`: Called when plugin is unloaded

## Error Handling

If a plugin fails to load or encounters an error during execution:
- A warning will be displayed
- Training will continue without the plugin
- Other plugins will not be affected

## Examples

See `example_plugin.py` for a complete example of a custom plugin.
