"""
Tiny GPT model (5M parameters)
Optimized for prototyping and small projects
"""

from .gpt import create_gpt_model


def create_tiny_model(model_config=None):
    """
    Create tiny GPT model with 5M parameters
    
    Architecture:
    - 4 layers
    - 4 attention heads
    - 256 hidden dimension
    - 10K vocabulary (or actual vocab size from tokenizer)
    - 512 max sequence length
    
    Hardware Requirements:
    - CPU or basic GPU
    - 4GB RAM minimum
    - Training time: 5-15 minutes
    
    Note:
    - vocab_size should match your trained tokenizer's vocabulary size
    - The default value (10000) is overridden by load_model_from_config()
      to use the actual tokenizer vocabulary size
    - Mismatched vocab sizes cause poor generation quality
    """
    config = {
        'vocab_size': 10000,  # Will be overridden by actual tokenizer vocab
        'max_length': 512,
        'layers': 4,
        'heads': 4,
        'dim': 256,
        'dropout': 0.2,
    }
    
    # Override with provided config (e.g., actual vocab size from tokenizer)
    if model_config:
        config.update(model_config)
    
    return create_gpt_model(config)


if __name__ == '__main__':
    model = create_tiny_model()
    print(f"Tiny model created with {model.count_parameters():,} parameters")
