"""
Nano GPT model (500K parameters)
Perfect for learning and quick testing
"""

from .gpt import create_gpt_model


def create_nano_model(model_config=None):
    """
    Create nano GPT model with 500K parameters
    
    Architecture:
    - 3 layers
    - 4 attention heads
    - 128 hidden dimension
    - 5K vocabulary (or actual vocab size from tokenizer)
    - 256 max sequence length
    
    Hardware Requirements:
    - Any CPU
    - 2GB RAM minimum
    - Training time: 1-2 minutes
    
    Note:
    - vocab_size should match your trained tokenizer's vocabulary size
    - The default value (5000) is overridden by load_model_from_config()
      to use the actual tokenizer vocabulary size
    - Mismatched vocab sizes cause poor generation quality
    """
    config = {
        'vocab_size': 5000,  # Will be overridden by actual tokenizer vocab
        'max_length': 256,
        'layers': 3,
        'heads': 4,
        'dim': 128,
        'dropout': 0.1,
    }
    
    # Override with provided config (e.g., actual vocab size from tokenizer)
    if model_config:
        config.update(model_config)
    
    return create_gpt_model(config)


if __name__ == '__main__':
    model = create_nano_model()
    print(f"Nano model created with {model.count_parameters():,} parameters")
