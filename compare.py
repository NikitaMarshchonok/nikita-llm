#!/usr/bin/env python3
"""
Model Comparison Tool
Compare multiple model checkpoints side-by-side on the same validation set
"""

import argparse
import sys
import torch
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import load_model_from_config
from data import LLMDataset, create_dataloader
from tokenizers import Tokenizer


class ModelComparator:
    """
    Compares multiple model checkpoints
    """
    
    def __init__(
        self,
        checkpoint_paths: List[str],
        val_data_path: str,
        tokenizer_path: str,
        device: str = 'auto'
    ):
        """
        Initialize model comparator
        
        Args:
            checkpoint_paths: List of checkpoint paths to compare
            val_data_path: Path to validation data
            tokenizer_path: Path to tokenizer
            device: Device to use (auto/cuda/cpu)
        """
        self.checkpoint_paths = [Path(p) for p in checkpoint_paths]
        self.val_data_path = Path(val_data_path)
        self.tokenizer_path = Path(tokenizer_path)
        
        # Setup device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Validate paths
        for cp in self.checkpoint_paths:
            if not cp.exists():
                raise FileNotFoundError(f"Checkpoint not found: {cp}")
        
        if not self.val_data_path.exists():
            raise FileNotFoundError(f"Validation data not found: {val_data_path}")
        
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        
        print(f"\n{'='*70}")
        print(f"🔍 Model Comparison Tool")
        print(f"{'='*70}")
        print(f"\n📊 Comparing {len(self.checkpoint_paths)} models")
        print(f"📁 Validation data: {val_data_path}")
        print(f"💻 Device: {self.device.upper()}")
    
    def load_model(self, checkpoint_path: Path) -> tuple:
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
        
        Returns:
            Tuple of (model, checkpoint_info)
        """
        print(f"\n📦 Loading {checkpoint_path.name}...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model
        model = load_model_from_config()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Get checkpoint info
        info = {
            'step': checkpoint.get('step', 'unknown'),
            'epoch': checkpoint.get('epoch', 'unknown'),
            'loss': checkpoint.get('loss', 'unknown'),
        }
        
        print(f"✓ Loaded (step: {info['step']}, epoch: {info['epoch']})")
        
        return model, info
    
    def evaluate_model(self, model, model_name: str) -> Dict[str, Any]:
        """
        Evaluate model on validation set
        
        Args:
            model: Model to evaluate
            model_name: Name for display
        
        Returns:
            Dictionary of metrics
        """
        print(f"\n📈 Evaluating {model_name}...")
        
        # Create validation dataloader
        val_dataset = LLMDataset(str(self.val_data_path), max_length=512)
        val_loader = create_dataloader(val_dataset, batch_size=8, shuffle=False)
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids, labels=labels)
                loss = outputs['loss']
                
                total_loss += loss.item() * input_ids.size(0)
                total_tokens += input_ids.numel()
        
        avg_loss = total_loss / len(val_dataset)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens_evaluated': total_tokens
        }
        
        print(f"✓ Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        return metrics
    
    def generate_samples(
        self,
        models: List[tuple],
        prompts: List[str],
        max_length: int = 50,
        temperature: float = 0.8
    ) -> Dict[str, List[str]]:
        """
        Generate sample outputs from all models
        
        Args:
            models: List of (model, name) tuples
            prompts: List of prompts to generate from
            max_length: Maximum generation length
            temperature: Sampling temperature
        
        Returns:
            Dictionary mapping model names to generated texts
        """
        print(f"\n✨ Generating samples from {len(models)} models...")
        
        results = {}
        
        for model, name in models:
            model_results = []
            
            for prompt in prompts:
                # Encode prompt
                encoding = self.tokenizer.encode(prompt)
                input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=self.device)
                
                # Generate
                with torch.no_grad():
                    generated = model.generate(
                        input_ids,
                        max_new_tokens=max_length,
                        temperature=temperature
                    )
                
                # Decode
                generated_ids = generated[0, input_ids.size(1):].tolist()
                generated_text = self.tokenizer.decode(generated_ids)
                
                model_results.append(generated_text)
            
            results[name] = model_results
            print(f"✓ Generated {len(prompts)} samples for {name}")
        
        return results
    
    def compare(
        self,
        sample_prompts: List[str] = None,
        save_report: bool = True
    ) -> Dict[str, Any]:
        """
        Compare all models
        
        Args:
            sample_prompts: Optional list of prompts for generation
            save_report: Whether to save comparison report
        
        Returns:
            Comparison results
        """
        # Default prompts
        if sample_prompts is None:
            sample_prompts = [
                "Once upon a time",
                "The future of AI",
                "In a world where"
            ]
        
        # Load all models
        models = []
        model_infos = []
        
        for cp in self.checkpoint_paths:
            model, info = self.load_model(cp)
            models.append((model, cp.stem))
            model_infos.append(info)
        
        # Evaluate all models
        print(f"\n{'='*70}")
        print(f"📊 Evaluation Results")
        print(f"{'='*70}")
        
        all_metrics = {}
        for (model, name), info in zip(models, model_infos):
            metrics = self.evaluate_model(model, name)
            metrics.update(info)
            all_metrics[name] = metrics
        
        # Display metrics table
        self._display_metrics_table(all_metrics)
        
        # Generate samples
        print(f"\n{'='*70}")
        print(f"✨ Sample Generations")
        print(f"{'='*70}")
        
        sample_results = self.generate_samples(models, sample_prompts)
        self._display_samples(sample_prompts, sample_results)
        
        # Prepare comparison results
        results = {
            'timestamp': datetime.now().isoformat(),
            'models': list(all_metrics.keys()),
            'metrics': all_metrics,
            'samples': {
                'prompts': sample_prompts,
                'generations': sample_results
            }
        }
        
        # Save report
        if save_report:
            report_path = self._save_report(results)
            print(f"\n💾 Comparison report saved to: {report_path}")
        
        # Display winner
        self._display_winner(all_metrics)
        
        return results
    
    def _display_metrics_table(self, metrics: Dict[str, Dict[str, Any]]) -> None:
        """Display metrics in a formatted table"""
        headers = ['Model', 'Step', 'Epoch', 'Loss', 'Perplexity', 'Tokens']
        rows = []
        
        for name, m in metrics.items():
            rows.append([
                name,
                m.get('step', 'N/A'),
                m.get('epoch', 'N/A'),
                f"{m['loss']:.4f}",
                f"{m['perplexity']:.2f}",
                f"{m['tokens_evaluated']:,}"
            ])
        
        print(f"\n{tabulate(rows, headers=headers, tablefmt='grid')}")
    
    def _display_samples(
        self,
        prompts: List[str],
        results: Dict[str, List[str]]
    ) -> None:
        """Display sample generations"""
        for i, prompt in enumerate(prompts):
            print(f"\n📝 Prompt {i+1}: \"{prompt}\"")
            print(f"{'─'*70}")
            
            for model_name, generations in results.items():
                print(f"\n🤖 {model_name}:")
                print(f"   {generations[i][:200]}...")
    
    def _display_winner(self, metrics: Dict[str, Dict[str, Any]]) -> None:
        """Display the best performing model"""
        # Find model with lowest loss
        best_model = min(metrics.items(), key=lambda x: x[1]['loss'])
        
        print(f"\n{'='*70}")
        print(f"🏆 Best Model: {best_model[0]}")
        print(f"{'='*70}")
        print(f"   Loss: {best_model[1]['loss']:.4f}")
        print(f"   Perplexity: {best_model[1]['perplexity']:.2f}")
        print(f"   Step: {best_model[1].get('step', 'N/A')}")
    
    def _save_report(self, results: Dict[str, Any]) -> Path:
        """Save comparison report to logs directory"""
        # Create logs directory
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = logs_dir / f'comparison_{timestamp}.json'
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return report_path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Compare multiple model checkpoints side-by-side',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two models
  python compare.py checkpoints/model_v1.pt checkpoints/model_v2.pt
  
  # Compare three models with custom prompts
  python compare.py model1.pt model2.pt model3.pt \\
                   --prompts "Hello world" "The quick brown fox"
  
  # Compare on specific validation data
  python compare.py model1.pt model2.pt --val-data data/processed/val.pt
  
  # Use CPU for comparison
  python compare.py model1.pt model2.pt --device cpu

Output:
  • Evaluation metrics table (loss, perplexity)
  • Sample generations from each model
  • Best model recommendation
  • Comparison report saved to logs/
        """
    )
    
    # Required arguments
    parser.add_argument(
        'checkpoints',
        nargs='+',
        help='Paths to model checkpoints to compare (2 or more)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--val-data',
        type=str,
        default='data/processed/val.pt',
        help='Path to validation data (default: data/processed/val.pt)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='tokenizer/tokenizer.json',
        help='Path to tokenizer (default: tokenizer/tokenizer.json)'
    )
    parser.add_argument(
        '--prompts',
        nargs='+',
        help='Custom prompts for sample generation'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=50,
        help='Maximum generation length (default: 50)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device to use (default: auto)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save comparison report'
    )
    
    return parser.parse_args()


def main():
    """Main comparison function"""
    args = parse_args()
    
    # Validate number of checkpoints
    if len(args.checkpoints) < 2:
        print("❌ Error: At least 2 checkpoints required for comparison")
        print("   Example: python compare.py model1.pt model2.pt")
        sys.exit(1)
    
    try:
        # Create comparator
        comparator = ModelComparator(
            checkpoint_paths=args.checkpoints,
            val_data_path=args.val_data,
            tokenizer_path=args.tokenizer,
            device=args.device
        )
        
        # Run comparison
        results = comparator.compare(
            sample_prompts=args.prompts,
            save_report=not args.no_save
        )
        
        print(f"\n{'='*70}")
        print(f"✅ Comparison complete!")
        print(f"{'='*70}\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   • Check that all checkpoint files exist")
        print(f"   • Verify validation data is prepared: python data/prepare.py")
        print(f"   • Ensure tokenizer is trained: python tokenizer/train.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
