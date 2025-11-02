"""
Hugging Face Hub Plugin
Enables easy model and tokenizer upload to Hugging Face Hub
"""

from typing import Dict, Any, Optional
from pathlib import Path
from .base import Plugin


class HuggingFacePlugin(Plugin):
    """
    Hugging Face Hub integration plugin
    
    Provides functionality to upload models and tokenizers to the
    Hugging Face Hub for easy sharing and deployment.
    """
    
    def __init__(self, name: str = 'huggingface'):
        super().__init__(name)
        self.hub = None
        self.repo_id = None
        self.private = False
        self.auto_upload = False
        self.upload_interval = 0
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize HuggingFace plugin
        
        Args:
            config: Full configuration from llm.config.js
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from huggingface_hub import HfApi, create_repo
            self.hub = HfApi()
            self.create_repo = create_repo
        except ImportError:
            print(f"⚠️  HuggingFace plugin: huggingface-hub package not installed")
            print(f"   Install with: pip install huggingface-hub")
            return False
        
        # Get HuggingFace config
        hf_config = config.get('huggingface', {})
        
        # Get repository settings
        self.repo_id = hf_config.get('repo_id', None)
        self.private = hf_config.get('private', False)
        self.auto_upload = hf_config.get('auto_upload', False)
        self.upload_interval = hf_config.get('upload_interval', 0)
        
        if not self.repo_id:
            print(f"⚠️  HuggingFace plugin: No repo_id configured")
            print(f"   Add 'repo_id' to huggingface config in llm.config.js")
            return False
        
        # Check authentication
        try:
            # Try to get user info to verify authentication
            user_info = self.hub.whoami()
            username = user_info['name']
            
            print(f"✓ HuggingFace plugin initialized")
            print(f"  User: {username}")
            print(f"  Repository: {self.repo_id}")
            print(f"  Private: {self.private}")
            
            # Create repository if it doesn't exist
            try:
                self.create_repo(
                    repo_id=self.repo_id,
                    private=self.private,
                    exist_ok=True
                )
                print(f"  ✓ Repository ready")
            except Exception as e:
                print(f"  ⚠️  Could not create repository: {e}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  HuggingFace plugin: Not authenticated")
            print(f"   Error: {e}")
            print(f"   Login with: huggingface-cli login")
            return False
    
    def on_train_begin(self, trainer) -> None:
        """Called when training begins"""
        if not self.hub:
            return
        
        try:
            print(f"[{self.name}] Model will be uploaded to: https://huggingface.co/{self.repo_id}")
            
            # Create model card
            self._create_model_card(trainer)
            
        except Exception as e:
            print(f"⚠️  HuggingFace plugin error in on_train_begin: {e}")
    
    def on_checkpoint_save(self, trainer, checkpoint_path: str) -> None:
        """
        Called when a checkpoint is saved
        
        Args:
            trainer: Trainer instance
            checkpoint_path: Path to saved checkpoint
        """
        if not self.hub or not self.auto_upload:
            return
        
        # Only upload at specified intervals
        if self.upload_interval > 0:
            if trainer.global_step % self.upload_interval != 0:
                return
        
        try:
            print(f"[{self.name}] Uploading checkpoint to Hugging Face Hub...")
            
            # Upload checkpoint
            self.hub.upload_file(
                path_or_fileobj=checkpoint_path,
                path_in_repo=f"checkpoints/{Path(checkpoint_path).name}",
                repo_id=self.repo_id,
                repo_type="model"
            )
            
            print(f"[{self.name}] ✓ Checkpoint uploaded")
            
        except Exception as e:
            print(f"⚠️  HuggingFace plugin error in on_checkpoint_save: {e}")
    
    def on_train_end(self, trainer, final_metrics: Dict[str, Any]) -> None:
        """
        Called when training ends
        
        Args:
            trainer: Trainer instance
            final_metrics: Final training metrics
        """
        if not self.hub:
            return
        
        try:
            print(f"[{self.name}] Uploading final model to Hugging Face Hub...")
            
            # Update model card with final metrics
            self._update_model_card(trainer, final_metrics)
            
            print(f"[{self.name}] ✓ Training complete")
            print(f"[{self.name}] View model at: https://huggingface.co/{self.repo_id}")
            
        except Exception as e:
            print(f"⚠️  HuggingFace plugin error in on_train_end: {e}")
    
    def _create_model_card(self, trainer) -> None:
        """
        Create model card for the repository
        
        Args:
            trainer: Trainer instance
        """
        try:
            # Get model info
            model_params = trainer.model.count_parameters() if hasattr(trainer.model, 'count_parameters') else 'Unknown'
            
            # Create model card content
            model_card = f"""---
language: en
license: apache-2.0
tags:
- llm
- gpt
- create-llm
---

# {self.repo_id.split('/')[-1]}

This model was trained using [create-llm](https://github.com/theaniketgiri/create-llm).

## Model Details

- **Model Type:** Language Model
- **Parameters:** {model_params:,} if isinstance(model_params, int) else model_params
- **Training Framework:** PyTorch
- **Created with:** create-llm

## Training Details

Training is in progress. Final metrics will be updated upon completion.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{self.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{self.repo_id}")

# Generate text
inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Citation

If you use this model, please cite:

```bibtex
@misc{{{self.repo_id.replace('/', '-')},
  author = {{Your Name}},
  title = {{{self.repo_id.split('/')[-1]}}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{self.repo_id}}}}}
}}
```
"""
            
            # Upload model card
            self.hub.upload_file(
                path_or_fileobj=model_card.encode(),
                path_in_repo="README.md",
                repo_id=self.repo_id,
                repo_type="model"
            )
            
            print(f"[{self.name}] ✓ Model card created")
            
        except Exception as e:
            print(f"⚠️  Error creating model card: {e}")
    
    def _update_model_card(self, trainer, final_metrics: Dict[str, Any]) -> None:
        """
        Update model card with final metrics
        
        Args:
            trainer: Trainer instance
            final_metrics: Final training metrics
        """
        try:
            # Get model info
            model_params = trainer.model.count_parameters() if hasattr(trainer.model, 'count_parameters') else 'Unknown'
            
            # Format metrics
            metrics_str = "\n".join([f"- **{k}:** {v:.4f}" if isinstance(v, float) else f"- **{k}:** {v}" 
                                     for k, v in final_metrics.items()])
            
            # Create updated model card
            model_card = f"""---
language: en
license: apache-2.0
tags:
- llm
- gpt
- create-llm
---

# {self.repo_id.split('/')[-1]}

This model was trained using [create-llm](https://github.com/theaniketgiri/create-llm).

## Model Details

- **Model Type:** Language Model
- **Parameters:** {model_params:,} if isinstance(model_params, int) else model_params
- **Training Framework:** PyTorch
- **Created with:** create-llm

## Training Details

### Final Metrics

{metrics_str}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{self.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{self.repo_id}")

# Generate text
inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Citation

If you use this model, please cite:

```bibtex
@misc{{{self.repo_id.replace('/', '-')},
  author = {{Your Name}},
  title = {{{self.repo_id.split('/')[-1]}}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{self.repo_id}}}}}
}}
```
"""
            
            # Upload updated model card
            self.hub.upload_file(
                path_or_fileobj=model_card.encode(),
                path_in_repo="README.md",
                repo_id=self.repo_id,
                repo_type="model"
            )
            
            print(f"[{self.name}] ✓ Model card updated with final metrics")
            
        except Exception as e:
            print(f"⚠️  Error updating model card: {e}")
    
    def upload_model(self, model_path: str) -> bool:
        """
        Upload model to Hugging Face Hub
        
        Args:
            model_path: Path to model directory or file
        
        Returns:
            True if successful, False otherwise
        """
        if not self.hub:
            print(f"⚠️  HuggingFace plugin not initialized")
            return False
        
        try:
            print(f"[{self.name}] Uploading model from {model_path}...")
            
            # Upload model files
            self.hub.upload_folder(
                folder_path=model_path,
                repo_id=self.repo_id,
                repo_type="model"
            )
            
            print(f"[{self.name}] ✓ Model uploaded successfully")
            print(f"[{self.name}] View at: https://huggingface.co/{self.repo_id}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Error uploading model: {e}")
            return False
    
    def upload_tokenizer(self, tokenizer_path: str) -> bool:
        """
        Upload tokenizer to Hugging Face Hub
        
        Args:
            tokenizer_path: Path to tokenizer file
        
        Returns:
            True if successful, False otherwise
        """
        if not self.hub:
            print(f"⚠️  HuggingFace plugin not initialized")
            return False
        
        try:
            print(f"[{self.name}] Uploading tokenizer from {tokenizer_path}...")
            
            # Upload tokenizer file
            self.hub.upload_file(
                path_or_fileobj=tokenizer_path,
                path_in_repo="tokenizer.json",
                repo_id=self.repo_id,
                repo_type="model"
            )
            
            print(f"[{self.name}] ✓ Tokenizer uploaded successfully")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Error uploading tokenizer: {e}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup HuggingFace resources"""
        if self.hub:
            print(f"[{self.name}] HuggingFace plugin cleanup complete")
