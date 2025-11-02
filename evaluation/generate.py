#!/usr/bin/env python3
"""
Text generation script
Generates text using trained model with various sampling strategies
"""

import argparse
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import load_model_from_config
from tokenizers import Tokenizer


class TextGenerator:
    """
    Text generator with multiple sampling strategies
    """
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1
    ):
        """
        Generate text with various sampling strategies
        
        Args:
            prompt: Starting text
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling (1.0 = disabled)
            num_return_sequences: Number of sequences to generate
        
        Returns:
            List of generated texts
        """
        # Encode prompt
        if prompt:
            encoding = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=self.device)
        else:
            # Start with BOS token if available
            input_ids = torch.tensor([[0]], dtype=torch.long, device=self.device)
        
        # Generate multiple sequences
        generated_sequences = []
        
        for _ in range(num_return_sequences):
            generated = self._generate_sequence(
                input_ids.clone(),
                max_length,
                temperature,
                top_k,
                top_p
            )
            generated_sequences.append(generated)
        
        return generated_sequences
    
    def _generate_sequence(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> str:
        """Generate a single sequence"""
        for _ in range(max_length):
            # Get model predictions
            outputs = self.model(input_ids)
            logits = outputs['logits']
            
            # Get logits for last token
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Check for EOS token (assuming token 2 is EOS)
            if next_token.item() == 2:
                break
        
        # Decode
        generated_ids = input_ids[0].tolist()
        generated_text = self.tokenizer.decode(generated_ids)
        
        return generated_text


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate text with LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with prompt
  python evaluation/generate.py --checkpoint checkpoints/final.pt --prompt "Once upon a time"
  
  # Higher temperature for more randomness
  python evaluation/generate.py --checkpoint checkpoints/final.pt --prompt "Hello" --temperature 1.5
  
  # Use top-k sampling
  python evaluation/generate.py --checkpoint checkpoints/final.pt --prompt "The" --top-k 40
  
  # Generate multiple sequences
  python evaluation/generate.py --checkpoint checkpoints/final.pt --prompt "AI is" --num-sequences 3
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='',
        help='Starting prompt for generation'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=100,
        help='Maximum tokens to generate (default: 100)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature, higher = more random (default: 1.0)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k sampling, 0 = disabled (default: 50)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Nucleus sampling, 1.0 = disabled (default: 0.95)'
    )
    parser.add_argument(
        '--num-sequences',
        type=int,
        default=1,
        help='Number of sequences to generate (default: 1)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use (default: auto)'
    )
    
    return parser.parse_args()


def main():
    """Main generation function"""
    args = parse_args()
    
    print("=" * 60)
    print("✨ Text Generation")
    print("=" * 60)
    
    try:
        # Setup device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
        
        print(f"\nDevice: {device}")
        
        # Load checkpoint
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        # Load model
        model = load_model_from_config()
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from step {checkpoint['step']}")
        
        # Load tokenizer
        tokenizer_path = Path('tokenizer/tokenizer.json')
        if not tokenizer_path.exists():
            print("❌ Tokenizer not found!")
            print("   Please train tokenizer first: python tokenizer/train.py --data data/raw/sample.txt")
            sys.exit(1)
        
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print("✓ Tokenizer loaded")
        
        # Create generator
        generator = TextGenerator(model, tokenizer, device)
        
        # Display settings
        print("\n" + "-" * 60)
        print("Generation Settings:")
        print(f"  Prompt: '{args.prompt}'")
        print(f"  Max length: {args.max_length}")
        print(f"  Temperature: {args.temperature}")
        print(f"  Top-k: {args.top_k}")
        print(f"  Top-p: {args.top_p}")
        print(f"  Sequences: {args.num_sequences}")
        print("-" * 60)
        
        # Generate
        print("\nGenerating...\n")
        generated_texts = generator.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_sequences
        )
        
        # Display results
        print("=" * 60)
        print("Generated Text:")
        print("=" * 60)
        
        for i, text in enumerate(generated_texts, 1):
            if args.num_sequences > 1:
                print(f"\n[Sequence {i}]")
            print(text)
            if i < len(generated_texts):
                print("\n" + "-" * 60)
        
        print("\n" + "=" * 60)
        print("\n✅ Generation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
