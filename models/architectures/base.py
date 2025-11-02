"""
Base GPT model (1B parameters)
Optimized for multi-GPU training with high quality
"""

from .gpt import create_gpt_model


def create_base_model(model_config=None):
    """
    Create base GPT model with 1B parameters
    
    Architecture:
    - 24 layers
    - 16 attention heads
    - 1536 hidden dimension
    - 50K vocabulary (or actual vocab size from tokenizer)
    - 2048 max sequence length
    
    Hardware Requirements:
    - NVIDIA A100 (40GB) or 2x RTX 4090
    - 64GB RAM minimum
    - Training time: 1-3 days
    
    Note:
    - vocab_size should match your trained tokenizer's vocabulary size
    - The default value (50000) is overridden by load_model_from_config()
      to use the actual tokenizer vocabulary size
    - Mismatched vocab sizes cause poor generation quality
    """
    config = {
        'vocab_size': 50000,  # Will be overridden by actual tokenizer vocab
        'max_length': 2048,
        'layers': 24,
        'heads': 16,
        'dim': 1536,
        'dropout': 0.1,
    }
    
    # Override with provided config (e.g., actual vocab size from tokenizer)
    if model_config:
        config.update(model_config)
    
    return create_gpt_model(config)


if __name__ == '__main__':
    model = create_base_model()
    print(f"Base model created with {model.count_parameters():,} parameters")
