"""
Pytest configuration and fixtures
Provides common fixtures for testing
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'model': {
            'type': 'gpt',
            'vocab_size': 1000,
            'max_length': 128,
            'layers': 2,
            'heads': 2,
            'dim': 64,
            'dropout': 0.1,
        },
        'training': {
            'batch_size': 2,
            'learning_rate': 1e-3,
            'max_steps': 10,
            'eval_interval': 5,
            'save_interval': 5,
        },
        'data': {
            'max_length': 128,
            'stride': 64,
        },
    }


@pytest.fixture
def sample_text():
    """Sample text data for testing"""
    return "This is a test sentence. " * 100


@pytest.fixture
def device():
    """Get device for testing (CPU by default)"""
    return 'cpu'


@pytest.fixture
def small_model(sample_config, device):
    """Create a small model for testing"""
    from models.architectures.gpt import create_gpt_model
    model = create_gpt_model(sample_config['model'])
    model.to(device)
    return model


@pytest.fixture
def sample_batch(device):
    """Create a sample batch for testing"""
    batch_size = 2
    seq_len = 32
    vocab_size = 1000
    
    return {
        'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        'labels': torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
    }
