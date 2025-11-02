"""
Model configuration management
Loads model config from llm.config.js with validation and hardware checks
"""

import json
import subprocess
import platform
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""
    pass


class ConfigLoader:
    """
    Configuration loader for LLM training
    Loads and validates llm.config.js file
    """
    
    def __init__(self, config_path: str = 'llm.config.js'):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> dict:
        """Load JavaScript config file using Node.js"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Make sure you're running from the project root directory."
            )
        
        # Use Node.js to parse the config file
        js_code = f"""
        const config = require('./{self.config_path}');
        console.log(JSON.stringify(config));
        """
        
        try:
            result = subprocess.run(
                ['node', '-e', js_code],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.config_path.parent
            )
            config = json.loads(result.stdout)
            return config
        except FileNotFoundError:
            raise RuntimeError(
                "Node.js not found. Please install Node.js to load config files.\n"
                "Visit: https://nodejs.org/"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to load config file:\n{e.stderr}\n"
                f"Check your llm.config.js for syntax errors."
            )
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse config as JSON: {e}")
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate model config
        self._validate_model_config()
        
        # Validate training config
        self._validate_training_config()
        
        # Validate data config
        self._validate_data_config()
        
        # Validate tokenizer config
        self._validate_tokenizer_config()
        
        # Check hardware compatibility
        self._check_hardware_compatibility()
    
    def _validate_model_config(self):
        """Validate model configuration"""
        model = self.config.get('model', {})
        
        if not model:
            raise ConfigValidationError("Missing 'model' section in config")
        
        # Check required fields
        required = ['type', 'vocab_size', 'max_length', 'layers', 'heads', 'dim']
        for field in required:
            if field not in model:
                raise ConfigValidationError(f"Missing required field: model.{field}")
        
        # Validate types
        if model['type'] not in ['gpt', 'bert', 't5']:
            raise ConfigValidationError(
                f"Invalid model type: {model['type']}. "
                f"Must be one of: gpt, bert, t5"
            )
        
        # Validate dimensions
        if model['dim'] % model['heads'] != 0:
            raise ConfigValidationError(
                f"model.dim ({model['dim']}) must be divisible by "
                f"model.heads ({model['heads']})"
            )
        
        # Validate positive values
        for field in ['vocab_size', 'max_length', 'layers', 'heads', 'dim']:
            if model[field] <= 0:
                raise ConfigValidationError(f"model.{field} must be positive")
        
        # Validate dropout
        dropout = model.get('dropout', 0.1)
        if not 0 <= dropout < 1:
            raise ConfigValidationError(
                f"model.dropout must be between 0 and 1, got {dropout}"
            )
    
    def _validate_training_config(self):
        """Validate training configuration"""
        training = self.config.get('training', {})
        
        if not training:
            raise ConfigValidationError("Missing 'training' section in config")
        
        # Validate positive values
        positive_fields = [
            'batch_size', 'learning_rate', 'max_steps',
            'eval_interval', 'save_interval', 'gradient_clip'
        ]
        for field in positive_fields:
            if field in training and training[field] <= 0:
                raise ConfigValidationError(f"training.{field} must be positive")
        
        # Validate optimizer
        optimizer = training.get('optimizer', 'adamw')
        if optimizer not in ['adamw', 'adam', 'sgd']:
            raise ConfigValidationError(
                f"Invalid optimizer: {optimizer}. "
                f"Must be one of: adamw, adam, sgd"
            )
        
        # Validate warmup steps
        warmup = training.get('warmup_steps', 0)
        if warmup < 0:
            raise ConfigValidationError("training.warmup_steps must be non-negative")
    
    def _validate_data_config(self):
        """Validate data configuration"""
        data = self.config.get('data', {})
        
        if not data:
            raise ConfigValidationError("Missing 'data' section in config")
        
        # Validate max_length and stride
        max_length = data.get('max_length', 512)
        stride = data.get('stride', 256)
        
        if max_length <= 0:
            raise ConfigValidationError("data.max_length must be positive")
        
        if stride <= 0:
            raise ConfigValidationError("data.stride must be positive")
        
        if stride > max_length:
            raise ConfigValidationError(
                f"data.stride ({stride}) cannot be greater than "
                f"data.max_length ({max_length})"
            )
        
        # Validate val_split
        val_split = data.get('val_split', 0.1)
        if not 0 <= val_split < 1:
            raise ConfigValidationError(
                f"data.val_split must be between 0 and 1, got {val_split}"
            )
    
    def _validate_tokenizer_config(self):
        """Validate tokenizer configuration"""
        tokenizer = self.config.get('tokenizer', {})
        
        if not tokenizer:
            raise ConfigValidationError("Missing 'tokenizer' section in config")
        
        # Validate tokenizer type
        tok_type = tokenizer.get('type', 'bpe')
        if tok_type not in ['bpe', 'wordpiece', 'unigram']:
            raise ConfigValidationError(
                f"Invalid tokenizer type: {tok_type}. "
                f"Must be one of: bpe, wordpiece, unigram"
            )
        
        # Validate vocab_size
        vocab_size = tokenizer.get('vocab_size', 32000)
        if vocab_size <= 0:
            raise ConfigValidationError("tokenizer.vocab_size must be positive")
    
    def _check_hardware_compatibility(self):
        """Check if hardware is compatible with config"""
        import torch
        
        model = self.config.get('model', {})
        training = self.config.get('training', {})
        
        # Estimate memory requirements (rough approximation)
        params = self._estimate_parameters()
        batch_size = training.get('batch_size', 32)
        max_length = model.get('max_length', 512)
        
        # Rough memory estimate in GB
        # params * 4 bytes (fp32) + activations + gradients + optimizer states
        memory_gb = (params * 4 * 4) / (1024 ** 3)  # 4x for gradients, optimizer
        memory_gb += (batch_size * max_length * model.get('dim', 512) * 4) / (1024 ** 3)
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            if memory_gb > 8:
                print(
                    f"⚠️  Warning: No GPU detected. Model requires ~{memory_gb:.1f}GB memory.\n"
                    f"   Training on CPU will be very slow. Consider using a smaller model."
                )
        else:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if memory_gb > gpu_memory * 0.8:  # Leave 20% headroom
                print(
                    f"⚠️  Warning: Model may not fit in GPU memory.\n"
                    f"   Estimated: {memory_gb:.1f}GB, Available: {gpu_memory:.1f}GB\n"
                    f"   Consider: reducing batch_size, enabling mixed_precision, "
                    f"or using gradient_accumulation"
                )
    
    def _estimate_parameters(self) -> int:
        """Estimate number of model parameters"""
        model = self.config.get('model', {})
        
        vocab_size = model.get('vocab_size', 32000)
        max_length = model.get('max_length', 512)
        layers = model.get('layers', 6)
        dim = model.get('dim', 384)
        
        # Rough parameter count
        # Embeddings: vocab_size * dim + max_length * dim
        # Each layer: 4 * dim^2 (attention) + 8 * dim^2 (FFN)
        # Output: dim * vocab_size (tied with input embedding)
        
        embeddings = vocab_size * dim + max_length * dim
        per_layer = 12 * dim * dim
        total = embeddings + layers * per_layer
        
        return total
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config.get('training', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.config.get('data', {})
    
    def get_tokenizer_config(self) -> Dict[str, Any]:
        """Get tokenizer configuration"""
        return self.config.get('tokenizer', {})
    
    def get_checkpoint_config(self) -> Dict[str, Any]:
        """Get checkpoint configuration"""
        return self.config.get('checkpoints', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config.get('logging', {})
    
    def get_plugins(self) -> list:
        """Get enabled plugins"""
        return self.config.get('plugins', [])


def get_tokenizer_vocab_size(tokenizer_path: Path) -> Optional[int]:
    """
    Read vocabulary size from tokenizer.json
    
    Args:
        tokenizer_path: Path to tokenizer.json file
        
    Returns:
        Vocabulary size if tokenizer exists and is valid, None otherwise
    """
    if not tokenizer_path.exists():
        return None
    
    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
            
        # Extract vocab from tokenizer data
        if 'model' not in tokenizer_data:
            raise ValueError("Invalid tokenizer format: missing 'model' key")
        
        if 'vocab' not in tokenizer_data['model']:
            raise ValueError("Invalid tokenizer format: missing 'vocab' key in model")
        
        vocab_size = len(tokenizer_data['model']['vocab'])
        return vocab_size
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Corrupted tokenizer file: {e}")
    except Exception as e:
        raise ValueError(f"Error reading tokenizer: {e}")


def load_model_from_config(config_path: str = 'llm.config.js'):
    """Load model based on config file"""
    from .architectures import nano, tiny, small, base, gpt
    import json
    from pathlib import Path
    
    config = ConfigLoader(config_path)
    model_config = config.get_model_config()
    
    # Auto-detect vocab size from tokenizer if available
    tokenizer_path = Path('tokenizer/tokenizer.json')
    config_vocab_size = model_config.get('vocab_size', 32000)
    
    try:
        actual_vocab_size = get_tokenizer_vocab_size(tokenizer_path)
        
        if actual_vocab_size is not None:
            if actual_vocab_size != config_vocab_size:
                print(f"\n⚠️  Vocab size mismatch detected!")
                print(f"   Config: {config_vocab_size:,} | Tokenizer: {actual_vocab_size:,}")
                print(f"   Using actual tokenizer vocab size: {actual_vocab_size:,}")
                model_config['vocab_size'] = actual_vocab_size
            else:
                print(f"✓ Vocab size: {actual_vocab_size:,}")
        else:
            print(f"⚠️  Tokenizer not found, using config vocab size: {config_vocab_size:,}")
            
    except ValueError as e:
        print(f"⚠️  Error reading tokenizer: {e}")
        print(f"   Using config vocab size: {config_vocab_size:,}")
    
    # Get model size
    size = model_config.get('size', 'small')
    
    # Create model based on size
    if size == 'nano':
        return nano.create_nano_model(model_config)
    elif size == 'tiny':
        return tiny.create_tiny_model(model_config)
    elif size == 'small':
        return small.create_small_model(model_config)
    elif size == 'base':
        return base.create_base_model(model_config)
    elif size == 'custom':
        return gpt.create_gpt_model(model_config)
    else:
        raise ValueError(f"Unknown model size: {size}")


if __name__ == '__main__':
    # Test config loading
    try:
        config = ConfigLoader()
        print("✓ Config loaded successfully")
        print(f"  Model: {config.get('model.type')} ({config.get('model.size')})")
        print(f"  Parameters: ~{config._estimate_parameters() / 1_000_000:.0f}M")
        print(f"  Batch size: {config.get('training.batch_size')}")
        print(f"  Max steps: {config.get('training.max_steps')}")
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
