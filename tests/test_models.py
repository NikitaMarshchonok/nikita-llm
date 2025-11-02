"""
Tests for model architectures
"""

import pytest
import torch
from models.architectures.gpt import GPTModel, GPTConfig, create_gpt_model


class TestGPTModel:
    """Tests for GPT model"""
    
    def test_model_creation(self, sample_config):
        """Test model can be created"""
        model = create_gpt_model(sample_config['model'])
        assert model is not None
        assert isinstance(model, GPTModel)
    
    def test_model_forward(self, small_model, sample_batch):
        """Test model forward pass"""
        outputs = small_model(
            sample_batch['input_ids'],
            labels=sample_batch['labels']
        )
        
        assert 'logits' in outputs
        assert 'loss' in outputs
        assert outputs['logits'].shape == (*sample_batch['input_ids'].shape, small_model.config.vocab_size)
    
    def test_model_generation(self, small_model, device):
        """Test model can generate text"""
        input_ids = torch.randint(0, 100, (1, 10), device=device)
        generated = small_model.generate(input_ids, max_new_tokens=20)
        
        assert generated.shape[1] > input_ids.shape[1]
        assert generated.shape[1] <= input_ids.shape[1] + 20
    
    def test_parameter_count(self, small_model):
        """Test parameter counting"""
        param_count = small_model.count_parameters()
        assert param_count > 0
        assert isinstance(param_count, int)
    
    def test_sequence_length_at_boundary(self, small_model, device):
        """Test model forward pass with sequence at max_length boundary"""
        max_length = small_model.config.max_length
        vocab_size = small_model.config.vocab_size
        
        # Create input exactly at max_length
        input_ids = torch.randint(0, vocab_size, (2, max_length), device=device)
        labels = torch.randint(0, vocab_size, (2, max_length), device=device)
        
        # Should not raise IndexError
        outputs = small_model(input_ids, labels=labels)
        
        assert 'logits' in outputs
        assert 'loss' in outputs
        assert outputs['logits'].shape[1] == max_length
    
    def test_sequence_length_exceeds_max(self, small_model, device):
        """Test model forward pass with sequence exceeding max_length"""
        max_length = small_model.config.max_length
        vocab_size = small_model.config.vocab_size
        
        # Create input longer than max_length
        long_length = max_length + 50
        input_ids = torch.randint(0, vocab_size, (2, long_length), device=device)
        labels = torch.randint(0, vocab_size, (2, long_length), device=device)
        
        # Should not raise IndexError (should truncate)
        outputs = small_model(input_ids, labels=labels)
        
        assert 'logits' in outputs
        assert 'loss' in outputs
        # Output should be truncated to max_length
        assert outputs['logits'].shape[1] == max_length
    
    def test_no_index_error_on_long_sequence(self, small_model, device):
        """Test that no IndexError is raised when sequence exceeds max_length"""
        max_length = small_model.config.max_length
        vocab_size = small_model.config.vocab_size
        
        # Create very long input
        very_long_length = max_length * 2
        input_ids = torch.randint(0, vocab_size, (1, very_long_length), device=device)
        
        # Should not raise IndexError
        try:
            outputs = small_model(input_ids)
            assert outputs is not None
        except IndexError:
            pytest.fail("IndexError raised for long sequence - truncation not working")


@pytest.mark.parametrize("template", ["tiny", "small", "base"])
def test_template_models(template):
    """Test that template models can be created"""
    if template == "tiny":
        from models.architectures.tiny import create_tiny_model
        model = create_tiny_model()
    elif template == "small":
        from models.architectures.small import create_small_model
        model = create_small_model()
    elif template == "base":
        from models.architectures.base import create_base_model
        model = create_base_model()
    
    assert model is not None
    assert model.count_parameters() > 0
