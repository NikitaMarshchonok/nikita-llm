#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training script
Trains the LLM model with full configuration support
"""

import argparse
import sys
import os
import json
import torch
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ConfigLoader, load_model_from_config
from data import LLMDataset, create_dataloader
from training import Trainer
from training.callbacks import CheckpointCallback, LoggingCallback, CheckpointManager


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train LLM model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python training/train.py
  
  # Train for specific number of steps
  python training/train.py --max-steps 500
  
  # Resume from checkpoint
  python training/train.py --resume checkpoints/checkpoint-5000.pt
  
  # Use custom config
  python training/train.py --config my-config.js
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='llm.config.js',
        help='Path to config file (default: llm.config.js)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume from checkpoint path'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to train on (default: auto)'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        help='Maximum training steps (overrides config)'
    )
    
    return parser.parse_args()


def setup_device(device_arg: str) -> str:
    """Setup training device"""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    
    return device


def load_data(config: ConfigLoader, device: str):
    """Load training and validation data"""
    data_config = config.get_data_config()
    training_config = config.get_training_config()
    
    # Check if processed data exists
    train_path = Path('data/processed/train.pt')
    val_path = Path('data/processed/val.pt')
    
    if not train_path.exists():
        print("❌ Training data not found!")
        print("   Please run: python data/prepare.py")
        sys.exit(1)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = LLMDataset(
        str(train_path),
        max_length=data_config.get('max_length', 512)
    )
    
    val_dataset = None
    if val_path.exists():
        val_dataset = LLMDataset(
            str(val_path),
            max_length=data_config.get('max_length', 512)
        )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=training_config.get('batch_size', 32),
        shuffle=data_config.get('shuffle', True),
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=(device == 'cuda')
    )
    
    val_loader = None
    if val_dataset:
        val_loader = create_dataloader(
            val_dataset,
            batch_size=training_config.get('batch_size', 32),
            shuffle=False,
            num_workers=0,
            pin_memory=(device == 'cuda')
        )
    
    print(f"✓ Loaded {len(train_dataset)} training examples")
    if val_dataset:
        print(f"✓ Loaded {len(val_dataset)} validation examples")
    
    return train_loader, val_loader


def create_callbacks(config: ConfigLoader):
    """Create training callbacks"""
    callbacks = []
    
    # Checkpoint callback
    checkpoint_config = config.get_checkpoint_config()
    training_config = config.get_training_config()
    
    checkpoint_callback = CheckpointCallback(
        checkpoint_dir='checkpoints',
        save_interval=training_config.get('save_interval', 5000),
        save_total_limit=checkpoint_config.get('save_total_limit', 3),
        save_best=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Logging callback
    logging_config = config.get_logging_config()
    logging_callback = LoggingCallback(
        log_interval=logging_config.get('log_interval', 100),
        log_dir=logging_config.get('log_dir', 'logs'),
        verbose=True,
        use_tensorboard=logging_config.get('tensorboard', True)
    )
    callbacks.append(logging_callback)
    
    return callbacks


def main():
    """Main training function"""
    args = parse_args()
    
    print("=" * 60)
    print("🚀 LLM Training")
    print("=" * 60)
    
    try:
        # Load config
        print(f"\nLoading config from: {args.config}")
        config = ConfigLoader(args.config)
        print("✓ Config loaded successfully")
        
        # Setup device
        device = setup_device(args.device)
        print(f"\nDevice: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load model
        print("\nLoading model...")
        model = load_model_from_config(args.config)
        num_params = model.count_parameters()
        print(f"✓ Model loaded: {num_params:,} parameters")
        
        # Validate tokenizer and vocab size
        print("\nValidating model configuration...")
        tokenizer_path = Path('tokenizer/tokenizer.json')
        if not tokenizer_path.exists():
            print("❌ Tokenizer not found!")
            print("   Please train tokenizer first: python tokenizer/train.py --data data/raw/")
            sys.exit(1)
        
        try:
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
                tokenizer_vocab_size = len(tokenizer_data['model']['vocab'])
            
            model_vocab_size = model.config.vocab_size
            
            if tokenizer_vocab_size != model_vocab_size:
                print(f"❌ Vocabulary size mismatch!")
                print(f"   Model vocab size: {model_vocab_size:,}")
                print(f"   Tokenizer vocab size: {tokenizer_vocab_size:,}")
                print(f"\n   This will cause training to fail or produce poor results.")
                print(f"   The model was auto-corrected during loading, but there may be")
                print(f"   a configuration issue. Please verify llm.config.js matches your tokenizer.")
                sys.exit(1)
            
            print(f"✓ Vocabulary sizes match: {model_vocab_size:,}")
            
        except Exception as e:
            print(f"❌ Error validating tokenizer: {e}")
            sys.exit(1)
        
        # Load data
        print()
        train_loader, val_loader = load_data(config, device)
        
        # Check for potential overfitting
        num_examples = len(train_loader.dataset)
        params_per_example = num_params / num_examples
        if params_per_example > 1000:
            print(f"\n⚠️  WARNING: Model may be too large for dataset!")
            print(f"   Model: {num_params:,} parameters")
            print(f"   Data: {num_examples:,} examples")
            print(f"   Ratio: {params_per_example:,.0f} params/example")
            print(f"   Recommendation: Use smaller model or add more data\n")
        
        # Create callbacks
        callbacks = create_callbacks(config)
        
        # Override max_steps if provided via CLI
        if args.max_steps is not None:
            config.config['training']['max_steps'] = args.max_steps
            print(f"\nOverriding max_steps from CLI: {args.max_steps}")
        
        # Create trainer
        print("\nInitializing trainer...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.config,
            callbacks=callbacks,
            device=device
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            print(f"\nResuming from checkpoint: {args.resume}")
            checkpoint_manager = CheckpointManager()
            checkpoint_manager.load_checkpoint(args.resume, trainer)
        
        # Start training
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)
        trainer.train()
        
        print("\nTraining completed successfully!")
        
        # Post-training menu
        while True:
            choice = show_post_training_menu()
            
            if choice == 'continue':
                continue_training(trainer, config)
            elif choice == 'chat':
                launch_chat_interface()
            else:
                break
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def show_post_training_menu():
    """
    Display post-training options menu
    
    Returns:
        User's choice ('continue', 'chat', 'exit')
    """
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nWhat would you like to do next?")
    print("  1. Continue training (add more steps)")
    print("  2. Launch chat interface (test your model)")
    print("  3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            return 'continue'
        elif choice == '2':
            return 'chat'
        elif choice == '3':
            return 'exit'
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def continue_training(trainer, config):
    """
    Continue training with additional steps
    
    Args:
        trainer: Trainer instance
        config: Configuration object
    """
    try:
        additional_steps = input("\nHow many additional steps? (default: 1000): ").strip()
        
        if not additional_steps:
            additional_steps = 1000
        else:
            additional_steps = int(additional_steps)
        
        if additional_steps <= 0:
            print("Invalid number of steps. Must be positive.")
            return
        
        print(f"\nContinuing training for {additional_steps} more steps...")
        
        # Update max_steps
        current_step = trainer.global_step
        trainer.max_steps = current_step + additional_steps
        
        # Resume training
        print("\n" + "=" * 60)
        print("Resuming training...")
        print("=" * 60)
        trainer.train()
        
        print("\nAdditional training completed!")
        
    except ValueError:
        print("Invalid input. Please enter a number.")
    except KeyboardInterrupt:
        print("\n\nTraining cancelled.")


def launch_chat_interface():
    """Launch Gradio chat interface"""
    print("\nLaunching chat interface...")
    print("The interface will open in your browser.")
    print("Press Ctrl+C to stop the server.\n")
    
    try:
        from chat_interface import ChatInterface
        
        chat = ChatInterface()
        chat.load_model()
        chat.launch()
        
    except ImportError:
        print("ERROR: Gradio not installed.")
        print("Install with: pip install gradio")
    except KeyboardInterrupt:
        print("\n\nChat interface stopped.")
    except Exception as e:
        print(f"ERROR: Failed to launch chat interface: {e}")


if __name__ == '__main__':
    main()
