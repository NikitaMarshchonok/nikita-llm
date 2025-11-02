#!/usr/bin/env python3
"""
Tokenizer training script
Trains BPE, WordPiece, or Unigram tokenizers on your data
"""

import argparse
import sys
from pathlib import Path
from typing import List
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence


def load_text_files(data_path: str) -> List[str]:
    """Load all text files from directory or single file"""
    path = Path(data_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    files = []
    if path.is_file():
        files = [str(path)]
    elif path.is_dir():
        files = [str(f) for f in path.glob('**/*.txt')]
        if not files:
            raise ValueError(f"No .txt files found in {data_path}")
    else:
        raise ValueError(f"Invalid path: {data_path}")
    
    print(f"Found {len(files)} text file(s)")
    return files


def train_bpe_tokenizer(
    files: List[str],
    vocab_size: int,
    min_frequency: int,
    special_tokens: List[str]
) -> Tokenizer:
    """Train BPE (Byte Pair Encoding) tokenizer"""
    print("\nTraining BPE tokenizer...")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    
    # Set normalizer and pre-tokenizer
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    
    # Configure trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True
    )
    
    # Train
    tokenizer.train(files, trainer)
    
    return tokenizer


def train_wordpiece_tokenizer(
    files: List[str],
    vocab_size: int,
    min_frequency: int,
    special_tokens: List[str]
) -> Tokenizer:
    """Train WordPiece tokenizer"""
    print("\nTraining WordPiece tokenizer...")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
    
    # Set normalizer and pre-tokenizer
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    
    # Configure trainer
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True
    )
    
    # Train
    tokenizer.train(files, trainer)
    
    return tokenizer


def train_unigram_tokenizer(
    files: List[str],
    vocab_size: int,
    min_frequency: int,
    special_tokens: List[str]
) -> Tokenizer:
    """Train Unigram tokenizer"""
    print("\nTraining Unigram tokenizer...")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(Unigram())
    
    # Set normalizer and pre-tokenizer
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    
    # Configure trainer
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        unk_token="<unk>"
    )
    
    # Train
    tokenizer.train(files, trainer)
    
    return tokenizer


def display_statistics(tokenizer: Tokenizer, sample_text: str):
    """Display tokenizer statistics"""
    vocab_size = tokenizer.get_vocab_size()
    
    print(f"\n{'='*60}")
    print("Tokenizer Statistics")
    print(f"{'='*60}")
    print(f"Vocabulary size: {vocab_size:,}")
    
    # Test encoding
    encoding = tokenizer.encode(sample_text)
    tokens = encoding.tokens
    ids = encoding.ids
    
    print(f"\nSample encoding:")
    print(f"Text: {sample_text}")
    print(f"Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
    print(f"Token count: {len(tokens)}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train a tokenizer on your text data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train BPE tokenizer on single file
  python tokenizer/train.py --data data/raw/train.txt --type bpe
  
  # Train WordPiece tokenizer on directory
  python tokenizer/train.py --data data/raw/ --type wordpiece --vocab-size 50000
  
  # Train with custom special tokens
  python tokenizer/train.py --data data/raw/ --type unigram --special-tokens "<s>" "</s>" "<pad>"
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to training data (file or directory)'
    )
    parser.add_argument(
        '--type',
        type=str,
        default='bpe',
        choices=['bpe', 'wordpiece', 'unigram'],
        help='Tokenizer type (default: bpe)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=32000,
        help='Vocabulary size (default: 32000)'
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=2,
        help='Minimum token frequency (default: 2)'
    )
    parser.add_argument(
        '--special-tokens',
        nargs='+',
        default=['<pad>', '<unk>', '<s>', '</s>'],
        help='Special tokens (default: <pad> <unk> <s> </s>)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='tokenizer/tokenizer.json',
        help='Output path for trained tokenizer (default: tokenizer/tokenizer.json)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load text files
        print(f"Loading data from: {args.data}")
        files = load_text_files(args.data)
        
        # Calculate total size
        total_size = sum(Path(f).stat().st_size for f in files)
        print(f"Total data size: {total_size / (1024**2):.2f} MB")
        
        # Train tokenizer based on type
        if args.type == 'bpe':
            tokenizer = train_bpe_tokenizer(
                files, args.vocab_size, args.min_frequency, args.special_tokens
            )
        elif args.type == 'wordpiece':
            tokenizer = train_wordpiece_tokenizer(
                files, args.vocab_size, args.min_frequency, args.special_tokens
            )
        elif args.type == 'unigram':
            tokenizer = train_unigram_tokenizer(
                files, args.vocab_size, args.min_frequency, args.special_tokens
            )
        else:
            raise ValueError(f"Unknown tokenizer type: {args.type}")
        
        # Display statistics
        sample_text = "This is a sample text to test the tokenizer."
        display_statistics(tokenizer, sample_text)
        
        # Save tokenizer
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output_path))
        
        print(f"✓ Tokenizer saved to: {output_path}")
        print(f"\nNext steps:")
        print(f"  1. Prepare your data: python data/prepare.py")
        print(f"  2. Start training: python training/train.py")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
