"""
Small GPT model (100M parameters)
Optimized for single GPU training with good performance
"""

from .gpt import create_gpt_model


def create_small_model(model_config=None):
    """
    Create small GPT model with 100M parameters
    
    Architecture:
    - 12 layers
    - 12 attention heads
    - 768 hidden dimension
    - 32K vocabulary (or actual vocab size from tokenizer)
    - 1024 max sequence length
    
    Hardware Requirements:
    - NVIDIA RTX 3060 (12GB) or better
    - 16GB RAM minimum
    - Training time: 2-6 hours
    
    Note:
    - vocab_size should match your trained tokenizer's vocabulary size
    - The default value (32000) is overridden by load_model_from_config()
      to use the actual tokenizer vocabulary size
    - Mismatched vocab sizes cause poor generation quality
    """
    config = {
        'vocab_size': 32000,  # Will be overridden by actual tokenizer vocab
        'max_length': 1024,
        'layers': 12,
        'heads': 12,
        'dim': 768,
        'dropout': 0.1,
    }
    
    # Override with provided config (e.g., actual vocab size from tokenizer)
    if model_config:
        config.update(model_config)
    
    return create_gpt_model(config)


if __name__ == '__main__':
    model = create_small_model()
    print(f"Small model created with {model.count_parameters():,} parameters")
