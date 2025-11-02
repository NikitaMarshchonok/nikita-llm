"""
Integration tests for end-to-end workflows
"""

import pytest
import torch


@pytest.mark.integration
class TestTrainingWorkflow:
    """Test complete training workflow"""
    
    @pytest.mark.slow
    def test_minimal_training(self, small_model, sample_batch, device):
        """Test minimal training loop"""
        model = small_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training step
        model.train()
        outputs = model(sample_batch['input_ids'], labels=sample_batch['labels'])
        loss = outputs['loss']
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        assert loss.item() > 0
    
    @pytest.mark.slow
    def test_evaluation(self, small_model, sample_batch, device):
        """Test evaluation"""
        model = small_model
        model.eval()
        
        with torch.no_grad():
            outputs = model(sample_batch['input_ids'], labels=sample_batch['labels'])
            loss = outputs['loss']
        
        assert loss.item() > 0


@pytest.mark.integration
class TestDataPipeline:
    """Test data processing pipeline"""
    
    def test_data_loading(self, temp_dir, sample_text):
        """Test data can be loaded and processed"""
        # Create sample data file
        data_file = temp_dir / "sample.txt"
        data_file.write_text(sample_text)
        
        # Test that file exists and can be read
        assert data_file.exists()
        content = data_file.read_text()
        assert len(content) > 0


@pytest.mark.integration
class TestEvaluationWithSequenceLengths:
    """Test evaluation with various sequence lengths"""
    
    def test_evaluation_with_short_sequences(self, small_model, device):
        """Test evaluation with sequences shorter than max_length"""
        from data.dataset import LLMDataset, create_dataloader
        
        max_length = small_model.config.max_length
        vocab_size = small_model.config.vocab_size
        
        # Create dataset with short sequences
        short_length = max_length // 2
        num_samples = 10
        data = torch.randint(0, vocab_size, (num_samples, short_length))
        
        # Mock dataset
        class MockDataset:
            def __len__(self):
                return num_samples
            
            def __getitem__(self, idx):
                return {
                    'input_ids': data[idx],
                    'attention_mask': torch.ones(short_length),
                    'labels': data[idx]
                }
        
        dataset = MockDataset()
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False, max_length=max_length)
        
        # Evaluate
        small_model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = small_model(**batch)
                total_loss += outputs['loss'].item()
        
        assert total_loss > 0
    
    def test_evaluation_with_max_length_sequences(self, small_model, device):
        """Test evaluation with sequences equal to max_length"""
        from data.dataset import LLMDataset, create_dataloader
        
        max_length = small_model.config.max_length
        vocab_size = small_model.config.vocab_size
        
        # Create dataset with sequences at max_length
        num_samples = 10
        data = torch.randint(0, vocab_size, (num_samples, max_length))
        
        # Mock dataset
        class MockDataset:
            def __len__(self):
                return num_samples
            
            def __getitem__(self, idx):
                return {
                    'input_ids': data[idx],
                    'attention_mask': torch.ones(max_length),
                    'labels': data[idx]
                }
        
        dataset = MockDataset()
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False, max_length=max_length)
        
        # Evaluate - should not raise IndexError
        small_model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = small_model(**batch)
                total_loss += outputs['loss'].item()
        
        assert total_loss > 0
    
    def test_evaluation_with_long_sequences(self, small_model, device):
        """Test evaluation with sequences longer than max_length"""
        from data.dataset import LLMDataset, create_dataloader
        
        max_length = small_model.config.max_length
        vocab_size = small_model.config.vocab_size
        
        # Create dataset with sequences longer than max_length
        long_length = max_length + 50
        num_samples = 10
        data = torch.randint(0, vocab_size, (num_samples, long_length))
        
        # Mock dataset
        class MockDataset:
            def __len__(self):
                return num_samples
            
            def __getitem__(self, idx):
                return {
                    'input_ids': data[idx],
                    'attention_mask': torch.ones(long_length),
                    'labels': data[idx]
                }
        
        dataset = MockDataset()
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False, max_length=max_length)
        
        # Evaluate - should not raise IndexError (sequences should be truncated)
        small_model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                # Verify batch was truncated
                assert batch['input_ids'].shape[1] <= max_length
                outputs = small_model(**batch)
                total_loss += outputs['loss'].item()
        
        assert total_loss > 0
    
    def test_evaluation_completes_successfully(self, small_model, device):
        """Test that evaluation completes successfully with mixed sequence lengths"""
        from data.dataset import LLMDataset, create_dataloader
        
        max_length = small_model.config.max_length
        vocab_size = small_model.config.vocab_size
        
        # Create dataset with mixed sequence lengths
        num_samples = 20
        sequences = []
        for i in range(num_samples):
            # Vary sequence length from short to very long
            seq_len = max_length // 2 + (i * 10)
            sequences.append(torch.randint(0, vocab_size, (seq_len,)))
        
        # Mock dataset
        class MockDataset:
            def __len__(self):
                return num_samples
            
            def __getitem__(self, idx):
                seq = sequences[idx]
                return {
                    'input_ids': seq,
                    'attention_mask': torch.ones(len(seq)),
                    'labels': seq
                }
        
        dataset = MockDataset()
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False, max_length=max_length)
        
        # Evaluate - should complete without errors
        small_model.eval()
        batch_count = 0
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = small_model(**batch)
                total_loss += outputs['loss'].item()
                batch_count += 1
        
        assert batch_count > 0
        assert total_loss > 0
    
    def test_metrics_calculated_correctly_after_truncation(self, small_model, device):
        """Test that metrics are calculated correctly when sequences are truncated"""
        from data.dataset import LLMDataset, create_dataloader
        
        max_length = small_model.config.max_length
        vocab_size = small_model.config.vocab_size
        
        # Create dataset with long sequences
        long_length = max_length * 2
        num_samples = 8
        data = torch.randint(0, vocab_size, (num_samples, long_length))
        
        # Mock dataset
        class MockDataset:
            def __len__(self):
                return num_samples
            
            def __getitem__(self, idx):
                return {
                    'input_ids': data[idx],
                    'attention_mask': torch.ones(long_length),
                    'labels': data[idx]
                }
        
        dataset = MockDataset()
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False, max_length=max_length)
        
        # Evaluate and calculate metrics
        small_model.eval()
        losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = small_model(**batch)
                losses.append(outputs['loss'].item())
        
        # Verify metrics are valid
        assert len(losses) > 0
        assert all(loss > 0 for loss in losses)
        avg_loss = sum(losses) / len(losses)
        assert avg_loss > 0
        
        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        assert perplexity > 1.0
