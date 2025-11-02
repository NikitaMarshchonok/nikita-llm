#!/usr/bin/env python3
"""
Interactive chat interface
Chat with your trained LLM model in the terminal

Features:
- Interactive terminal session with conversation history
- Context maintenance across multiple turns
- Loading indicator during generation
- Multiple sampling strategies (temperature, top-k, top-p)
- Commands: exit, quit, clear, reset, help
"""

import argparse
import sys
import torch
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import load_model_from_config, ConfigLoader
from tokenizers import Tokenizer


class ChatSession:
    """
    Interactive chat session with LLM
    Maintains conversation context and generates responses
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda',
        max_context_length: int = 512,
        context_window: int = 10
    ):
        """
        Initialize chat session
        
        Args:
            model: The LLM model
            tokenizer: Tokenizer for encoding/decoding
            device: Device to run on ('cuda' or 'cpu')
            max_context_length: Maximum token length for context
            context_window: Number of conversation turns to keep
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_context_length = max_context_length
        self.context_window = context_window
        
        self.model.to(device)
        self.model.eval()
        
        # Conversation history
        self.context = []
    
    def generate_response(
        self,
        user_input: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> str:
        """
        Generate response to user input
        
        Args:
            user_input: User's message
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling (1.0 = disabled)
        
        Returns:
            Generated response text
        """
        # Add user input to context
        self.context.append(f"User: {user_input}")
        
        # Build context text
        context_text = "\n".join(self.context) + "\nAssistant:"
        
        # Encode context
        encoding = self.tokenizer.encode(context_text)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=self.device)
        
        # Trim context if too long
        if input_ids.size(1) > self.max_context_length:
            input_ids = input_ids[:, -self.max_context_length:]
        
        # Generate response
        with torch.no_grad():
            generated_ids = self._generate(
                input_ids,
                max_length,
                temperature,
                top_k,
                top_p
            )
        
        # Decode response (only new tokens)
        response_ids = generated_ids[0, input_ids.size(1):].tolist()
        response = self.tokenizer.decode(response_ids)
        
        # Clean up response (stop at next turn markers)
        response = response.split("User:")[0].strip()
        response = response.split("Assistant:")[0].strip()
        
        # Remove any trailing special tokens
        for stop_word in ["<|endoftext|>", "</s>", "<eos>"]:
            response = response.replace(stop_word, "")
        response = response.strip()
        
        # Add to context
        self.context.append(f"Assistant: {response}")
        
        # Trim context to keep only recent turns
        if len(self.context) > self.context_window * 2:  # 2 messages per turn
            self.context = self.context[-self.context_window * 2:]
        
        return response
    
    def _generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively with sampling
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
        
        Returns:
            Generated token IDs
        """
        for _ in range(max_length):
            # Crop to model's max length if needed
            input_ids_cond = input_ids
            if input_ids.size(1) > self.model.config.max_length:
                input_ids_cond = input_ids[:, -self.model.config.max_length:]
            
            # Forward pass
            outputs = self.model(input_ids_cond)
            logits = outputs['logits']
            
            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                indices_to_remove = next_token_logits < top_k_values[-1]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Check for EOS tokens (common EOS token IDs)
            if next_token.item() in [0, 1, 2, 3]:  # Common EOS/PAD tokens
                break
            
            # Check for newline (stop at double newline)
            if next_token.item() == 10:  # Newline
                break
        
        return input_ids
    
    def reset_context(self):
        """Clear conversation history"""
        self.context = []
    
    def get_context_length(self) -> int:
        """Get current context length in tokens"""
        context_text = "\n".join(self.context)
        encoding = self.tokenizer.encode(context_text)
        return len(encoding.ids)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Interactive chat with your trained LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands during chat:
  exit, quit    - Exit the chat session
  clear, reset  - Clear conversation history
  help          - Show help message
  
Generation parameters:
  --temperature  Controls randomness (0.1-2.0, default: 0.8)
                 Lower = more focused, Higher = more creative
  --top-k        Top-k sampling (default: 50)
  --top-p        Nucleus sampling (default: 0.95)
  --max-length   Maximum response length in tokens (default: 100)

Examples:
  # Start chat with default settings
  python chat.py --checkpoint checkpoints/final.pt
  
  # More creative responses
  python chat.py --checkpoint checkpoints/final.pt --temperature 1.2
  
  # More focused responses
  python chat.py --checkpoint checkpoints/final.pt --temperature 0.5 --top-k 20
  
  # Longer responses
  python chat.py --checkpoint checkpoints/final.pt --max-length 200
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (e.g., checkpoints/final.pt)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='llm.config.js',
        help='Path to config file (default: llm.config.js)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=100,
        help='Maximum response length in tokens (default: 100)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k sampling (default: 50, 0 to disable)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Nucleus sampling (default: 0.95, 1.0 to disable)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use (default: auto)'
    )
    parser.add_argument(
        '--context-window',
        type=int,
        default=10,
        help='Number of conversation turns to keep (default: 10)'
    )
    
    return parser.parse_args()


def main():
    """Main chat function"""
    args = parse_args()
    
    # Print header
    print("=" * 70)
    print("💬  Interactive Chat with LLM")
    print("=" * 70)
    
    try:
        # Setup device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
        
        print(f"\n📱 Device: {device.upper()}")
        if device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Load checkpoint
        print(f"\n📦 Loading checkpoint: {args.checkpoint}")
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {args.checkpoint}")
            print("\nAvailable checkpoints:")
            checkpoint_dir = Path('checkpoints')
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob('*.pt'))
                if checkpoints:
                    for cp in sorted(checkpoints):
                        print(f"   - {cp}")
                    print(f"\n💡 Try using: python chat.py --checkpoint {checkpoints[0]}")
                else:
                    print("   No checkpoints found")
                    print("\n💡 Train a model first: python training/train.py")
            sys.exit(1)
        
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
        
        # Load model
        print(f"\n🤖 Loading model...")
        try:
            config = ConfigLoader(args.config)
            model = load_model_from_config(args.config)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Get training info
            step = checkpoint.get('step', 'unknown')
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"   ✓ Model loaded (step: {step}, epoch: {epoch})")
            print(f"   ✓ Parameters: {model.count_parameters():,}")
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            sys.exit(1)
        
        # Load tokenizer
        print(f"\n📝 Loading tokenizer...")
        tokenizer_path = Path('tokenizer/tokenizer.json')
        if not tokenizer_path.exists():
            print("   ❌ Tokenizer not found!")
            print("   Please train tokenizer first:")
            print("   python tokenizer/train.py --data data/raw/sample.txt")
            sys.exit(1)
        
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"   ✓ Tokenizer loaded (vocab size: {tokenizer.get_vocab_size()})")
        
        # Create chat session
        max_context = config.get('model.max_length', 512)
        chat = ChatSession(
            model,
            tokenizer,
            device,
            max_context_length=max_context,
            context_window=args.context_window
        )
        
        # Display settings
        print("\n⚙️  Generation settings:")
        print(f"   Temperature: {args.temperature}")
        print(f"   Top-k: {args.top_k}")
        print(f"   Top-p: {args.top_p}")
        print(f"   Max length: {args.max_length} tokens")
        print(f"   Context window: {args.context_window} turns")
        
        # Display instructions
        print("\n" + "=" * 70)
        print("Chat started! Type your message and press Enter.")
        print("Commands: 'exit' or 'quit' to exit, 'clear' to reset, 'help' for help")
        print("=" * 70 + "\n")
        
        # Chat loop
        turn_count = 0
        while True:
            try:
                # Get user input
                user_input = input("\n\033[1;36mYou:\033[0m ").strip()
                
                # Check for commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!\n")
                    break
                
                if user_input.lower() in ['clear', 'reset']:
                    chat.reset_context()
                    turn_count = 0
                    print("\n✓ Context cleared\n")
                    continue
                
                if user_input.lower() == 'help':
                    print("\n📖 Commands:")
                    print("   exit, quit, q  - Exit chat")
                    print("   clear, reset   - Clear conversation history")
                    print("   help           - Show this message")
                    print("\n💡 Tips:")
                    print("   - The model maintains context across turns")
                    print("   - Use 'clear' if responses become incoherent")
                    print(f"   - Current context: {chat.get_context_length()} tokens")
                    continue
                
                if not user_input:
                    continue
                
                # Show loading indicator
                print("\n\033[1;32mAssistant:\033[0m ", end="", flush=True)
                print("⏳ Thinking...", end="", flush=True)
                
                # Generate response
                response = chat.generate_response(
                    user_input,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                
                # Clear loading indicator and print response
                print("\r\033[1;32mAssistant:\033[0m " + response)
                
                turn_count += 1
                
                # Show context info every 5 turns
                if turn_count % 5 == 0:
                    context_len = chat.get_context_length()
                    print(f"\n   ℹ️  Context: {context_len}/{max_context} tokens, {len(chat.context)//2} turns")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            
            except Exception as e:
                print(f"\n\n❌ Error generating response: {e}")
                print("   Try 'clear' to reset context or 'exit' to quit\n")
                continue
    
    except Exception as e:
        print(f"\n❌ Failed to start chat: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
