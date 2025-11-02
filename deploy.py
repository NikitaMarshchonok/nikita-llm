#!/usr/bin/env python3
"""
Model Deployment Script
Deploy trained models to various platforms (Hugging Face Hub, Replicate, etc.)
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


class DeploymentManager:
    """
    Manages model deployment to various platforms
    """
    
    def __init__(self, checkpoint_path: str, tokenizer_path: Optional[str] = None):
        """
        Initialize deployment manager
        
        Args:
            checkpoint_path: Path to model checkpoint
            tokenizer_path: Optional path to tokenizer
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None
        
        # Validate paths
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if self.tokenizer_path and not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    def deploy_to_huggingface(
        self,
        repo_id: str,
        private: bool = False,
        token: Optional[str] = None
    ) -> str:
        """
        Deploy model to Hugging Face Hub
        
        Args:
            repo_id: Repository ID (username/model-name)
            private: Whether to make repository private
            token: Optional HuggingFace token
        
        Returns:
            Model URL on success
        
        Raises:
            Exception: If deployment fails
        """
        try:
            from huggingface_hub import HfApi, create_repo, login
        except ImportError:
            raise ImportError(
                "huggingface-hub not installed.\n"
                "Install with: pip install huggingface-hub"
            )
        
        print("\n" + "=" * 60)
        print("🚀 Deploying to Hugging Face Hub")
        print("=" * 60)
        
        # Login if token provided
        if token:
            print("\n📝 Logging in with provided token...")
            login(token=token)
        
        # Initialize API
        api = HfApi()
        
        # Verify authentication
        try:
            user_info = api.whoami()
            username = user_info['name']
            print(f"\n✓ Authenticated as: {username}")
        except Exception as e:
            raise Exception(
                f"Authentication failed: {e}\n"
                f"Please login with: huggingface-cli login\n"
                f"Or provide token with --token argument"
            )
        
        # Create repository
        print(f"\n📦 Creating repository: {repo_id}")
        try:
            create_repo(
                repo_id=repo_id,
                private=private,
                exist_ok=True,
                repo_type="model"
            )
            print(f"✓ Repository ready")
        except Exception as e:
            raise Exception(f"Failed to create repository: {e}")
        
        # Upload model checkpoint
        print(f"\n⬆️  Uploading model checkpoint...")
        try:
            api.upload_file(
                path_or_fileobj=str(self.checkpoint_path),
                path_in_repo=f"pytorch_model.bin",
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"✓ Model checkpoint uploaded")
        except Exception as e:
            raise Exception(f"Failed to upload model: {e}")
        
        # Upload tokenizer if provided
        if self.tokenizer_path:
            print(f"\n⬆️  Uploading tokenizer...")
            try:
                api.upload_file(
                    path_or_fileobj=str(self.tokenizer_path),
                    path_in_repo="tokenizer.json",
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"✓ Tokenizer uploaded")
            except Exception as e:
                print(f"⚠️  Warning: Failed to upload tokenizer: {e}")
        
        # Create model card
        print(f"\n📄 Creating model card...")
        try:
            self._create_huggingface_model_card(api, repo_id)
            print(f"✓ Model card created")
        except Exception as e:
            print(f"⚠️  Warning: Failed to create model card: {e}")
        
        # Success
        model_url = f"https://huggingface.co/{repo_id}"
        print("\n" + "=" * 60)
        print("✅ Deployment successful!")
        print("=" * 60)
        print(f"\n🔗 Model URL: {model_url}")
        print(f"\n💡 Next steps:")
        print(f"   • View your model: {model_url}")
        print(f"   • Test inference: Use the Inference API on the model page")
        print(f"   • Share with others: Send them the model URL")
        
        return model_url
    
    def deploy_to_replicate(
        self,
        model_name: str,
        token: Optional[str] = None
    ) -> str:
        """
        Deploy model to Replicate
        
        Args:
            model_name: Model name on Replicate
            token: Optional Replicate API token
        
        Returns:
            Model URL on success
        
        Raises:
            Exception: If deployment fails
        """
        print("\n" + "=" * 60)
        print("🚀 Deploying to Replicate")
        print("=" * 60)
        
        # Check for Replicate token
        replicate_token = token or os.environ.get('REPLICATE_API_TOKEN')
        
        if not replicate_token:
            raise Exception(
                "Replicate API token not found.\n"
                "Set REPLICATE_API_TOKEN environment variable or use --token argument.\n"
                "Get your token at: https://replicate.com/account/api-tokens"
            )
        
        print("\n📝 Note: Replicate deployment requires:")
        print("   1. A Cog configuration file (cog.yaml)")
        print("   2. A predict.py file with model inference code")
        print("   3. Docker installed on your system")
        print("   4. Cog CLI tool installed")
        
        print("\n💡 To deploy to Replicate:")
        print("   1. Install Cog: https://github.com/replicate/cog")
        print("   2. Create cog.yaml and predict.py in your project")
        print("   3. Run: cog push r8.im/username/model-name")
        
        print("\n⚠️  Automatic Replicate deployment is not yet implemented.")
        print("   Please follow the manual steps above.")
        
        # For now, return a placeholder URL
        model_url = f"https://replicate.com/{model_name}"
        return model_url
    
    def _create_huggingface_model_card(self, api, repo_id: str) -> None:
        """
        Create a model card for Hugging Face Hub
        
        Args:
            api: HuggingFace API instance
            repo_id: Repository ID
        """
        model_card = f"""---
language: en
license: apache-2.0
tags:
- llm
- gpt
- create-llm
- pytorch
---

# {repo_id.split('/')[-1]}

This model was trained using [create-llm](https://github.com/theaniketgiri/create-llm).

## Model Description

A language model trained with create-llm framework.

## Usage

```python
import torch
from transformers import AutoTokenizer

# Load model
model = torch.load('pytorch_model.bin')
model.eval()

# Load tokenizer (if available)
try:
    tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
except:
    print("Tokenizer not available")

# Generate text
# Add your generation code here
```

## Training Details

- **Framework:** PyTorch
- **Tool:** create-llm
- **Deployment:** Hugging Face Hub

## Citation

```bibtex
@misc{{{repo_id.replace('/', '-')},
  author = {{Your Name}},
  title = {{{repo_id.split('/')[-1]}}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{repo_id}}}}}
}}
```
"""
        
        # Upload model card
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model"
        )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Deploy trained model to various platforms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Deployment Platforms:
  huggingface    Deploy to Hugging Face Hub (model hosting and sharing)
  replicate      Deploy to Replicate (API-based inference)

Examples:
  # Deploy to Hugging Face Hub
  python deploy.py --checkpoint checkpoints/final.pt --to huggingface --repo-id username/model-name
  
  # Deploy with tokenizer
  python deploy.py --checkpoint checkpoints/final.pt --tokenizer tokenizer/tokenizer.json \\
                   --to huggingface --repo-id username/model-name
  
  # Deploy private model
  python deploy.py --checkpoint checkpoints/final.pt --to huggingface \\
                   --repo-id username/model-name --private
  
  # Deploy to Replicate
  python deploy.py --checkpoint checkpoints/final.pt --to replicate --model-name username/model-name

Authentication:
  Hugging Face: Run 'huggingface-cli login' or use --token
  Replicate: Set REPLICATE_API_TOKEN environment variable or use --token
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        '--to',
        type=str,
        required=True,
        choices=['huggingface', 'replicate'],
        help='Deployment platform'
    )
    
    # Optional arguments
    parser.add_argument(
        '--tokenizer',
        type=str,
        help='Path to tokenizer file (optional)'
    )
    parser.add_argument(
        '--token',
        type=str,
        help='API token for authentication (optional)'
    )
    
    # Hugging Face specific
    parser.add_argument(
        '--repo-id',
        type=str,
        help='Repository ID for Hugging Face (username/model-name)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make Hugging Face repository private'
    )
    
    # Replicate specific
    parser.add_argument(
        '--model-name',
        type=str,
        help='Model name for Replicate (username/model-name)'
    )
    
    return parser.parse_args()


def main():
    """Main deployment function"""
    args = parse_args()
    
    print("=" * 60)
    print("🚀 Model Deployment Tool")
    print("=" * 60)
    
    try:
        # Create deployment manager
        manager = DeploymentManager(
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer
        )
        
        # Deploy based on platform
        if args.to == 'huggingface':
            # Validate Hugging Face arguments
            if not args.repo_id:
                print("\n❌ Error: --repo-id is required for Hugging Face deployment")
                print("   Example: --repo-id username/model-name")
                sys.exit(1)
            
            # Deploy to Hugging Face
            url = manager.deploy_to_huggingface(
                repo_id=args.repo_id,
                private=args.private,
                token=args.token
            )
            
        elif args.to == 'replicate':
            # Validate Replicate arguments
            if not args.model_name:
                print("\n❌ Error: --model-name is required for Replicate deployment")
                print("   Example: --model-name username/model-name")
                sys.exit(1)
            
            # Deploy to Replicate
            url = manager.deploy_to_replicate(
                model_name=args.model_name,
                token=args.token
            )
        
        print("\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   • Check that the checkpoint file exists")
        print("   • Verify the file path is correct")
        print("   • Make sure you're running from the project root")
        sys.exit(1)
        
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   • Install required packages: pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   • Check your internet connection")
        print("   • Verify your authentication credentials")
        print("   • Make sure you have permission to create repositories")
        print("   • Check the platform's status page for outages")
        sys.exit(1)


if __name__ == '__main__':
    main()
