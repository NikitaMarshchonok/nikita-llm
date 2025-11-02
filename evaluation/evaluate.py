#!/usr/bin/env python3
"""
Model evaluation script
Evaluates trained model on validation set with perplexity and metrics
"""

import argparse
import sys
import torch
import time
import json
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import load_model_from_config
from data import LLMDataset, create_dataloader


class Evaluator:
    """
    Model evaluator for computing metrics
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Log model configuration for diagnostics
        if hasattr(model, 'config'):
            print(f"Model Configuration:")
            print(f"  max_length: {model.config.max_length}")
            print(f"  vocab_size: {model.config.vocab_size}")
            if hasattr(model, 'position_embedding'):
                print(f"  Position embedding size: {model.position_embedding.num_embeddings}")
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """
        Evaluate model on dataset
        
        Returns:
            dict with metrics: loss, perplexity, tokens_per_sec
        """
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        start_time = time.time()
        
        print("\nEvaluating...")
        for batch in tqdm(dataloader, desc="Evaluation"):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs['loss']
            
            # Accumulate
            total_loss += loss.item()
            total_tokens += batch['input_ids'].numel()
            num_batches += 1
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        tokens_per_sec = total_tokens / elapsed
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens_per_sec': tokens_per_sec,
            'total_tokens': total_tokens,
            'num_batches': num_batches,
            'elapsed_time': elapsed
        }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate LLM model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on validation set
  python evaluation/evaluate.py --checkpoint checkpoints/final.pt
  
  # Evaluate on custom data
  python evaluation/evaluate.py --checkpoint checkpoints/final.pt --data data/processed/test.pt
  
  # Save results to file
  python evaluation/evaluate.py --checkpoint checkpoints/final.pt --output results.json
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/val.pt',
        help='Path to evaluation data (default: data/processed/val.pt)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use (default: auto)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )
    
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: str):
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model
    model = load_model_from_config()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded from step {checkpoint['step']}")
    print(f"  Training loss: {checkpoint['loss']:.4f}")
    
    # Validate position embedding size matches config
    if hasattr(model, 'config') and hasattr(model, 'position_embedding'):
        config_max_length = model.config.max_length
        actual_max_length = model.position_embedding.num_embeddings
        
        if config_max_length != actual_max_length:
            print(f"\n⚠️  Position embedding size mismatch detected!")
            print(f"   Config max_length: {config_max_length} | Actual: {actual_max_length}")
            print(f"   Using actual position embedding size: {actual_max_length}")
            model.config.max_length = actual_max_length
    
    return model, checkpoint


def main():
    """Main evaluation function"""
    args = parse_args()
    
    print("=" * 60)
    print("📊 Model Evaluation")
    print("=" * 60)
    
    try:
        # Setup device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
        
        print(f"\nDevice: {device}")
        
        # Check data exists
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"❌ Data file not found: {args.data}")
            sys.exit(1)
        
        # Load checkpoint
        model, checkpoint = load_checkpoint(args.checkpoint, device)
        
        # Extract max_length from model configuration
        max_length = model.config.max_length if hasattr(model, 'config') else 512
        print(f"Using max_length: {max_length}")
        
        # Load data
        print(f"\nLoading data: {args.data}")
        dataset = LLMDataset(str(data_path))
        dataloader = create_dataloader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device == 'cuda'),
            max_length=max_length
        )
        print(f"✓ Loaded {len(dataset)} examples")
        
        # Create evaluator
        evaluator = Evaluator(model, device)
        
        # Evaluate with enhanced error handling
        try:
            metrics = evaluator.evaluate(dataloader)
        except IndexError as e:
            if "index out of range" in str(e):
                print("\n" + "=" * 60)
                print("❌ POSITION EMBEDDING INDEX ERROR")
                print("=" * 60)
                print(f"Model max_length: {max_length}")
                print(f"\nThis error occurs when validation sequences exceed the model's maximum length.")
                print(f"\nSolutions:")
                print(f"  1. Reprocess validation data with max_length={max_length}")
                print(f"  2. Increase model's max_length in config and retrain")
                print(f"  3. Check data preprocessing pipeline")
                print("=" * 60)
            raise
        
        # Display results
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Loss:           {metrics['loss']:.4f}")
        print(f"Perplexity:     {metrics['perplexity']:.2f}")
        print(f"Tokens/sec:     {metrics['tokens_per_sec']:.0f}")
        print(f"Total tokens:   {metrics['total_tokens']:,}")
        print(f"Batches:        {metrics['num_batches']}")
        print(f"Time:           {metrics['elapsed_time']:.2f}s")
        print("=" * 60)
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            results = {
                'checkpoint': args.checkpoint,
                'data': args.data,
                'metrics': metrics,
                'checkpoint_step': checkpoint['step'],
                'checkpoint_loss': checkpoint['loss']
            }
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✓ Results saved to: {output_path}")
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
