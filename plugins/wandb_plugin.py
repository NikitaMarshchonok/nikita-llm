"""
Weights & Biases (WandB) Plugin
Logs training metrics and artifacts to Weights & Biases
"""

from typing import Dict, Any, Optional
from .base import Plugin


class WandBPlugin(Plugin):
    """
    Weights & Biases integration plugin
    
    Logs training metrics, model artifacts, and system info to WandB.
    Requires wandb package to be installed and configured.
    """
    
    def __init__(self, name: str = 'wandb'):
        super().__init__(name)
        self.wandb = None
        self.run = None
        self.project_name = None
        self.run_name = None
        self.log_interval = 1
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize WandB plugin
        
        Args:
            config: Full configuration from llm.config.js
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            print(f"⚠️  WandB plugin: wandb package not installed")
            print(f"   Install with: pip install wandb")
            return False
        
        # Get WandB config
        wandb_config = config.get('wandb', {})
        
        # Get project and run names
        self.project_name = wandb_config.get('project', 'llm-training')
        self.run_name = wandb_config.get('run_name', None)
        self.log_interval = wandb_config.get('log_interval', 1)
        
        # Get entity (team/user)
        entity = wandb_config.get('entity', None)
        
        # Get tags
        tags = wandb_config.get('tags', [])
        
        # Get notes
        notes = wandb_config.get('notes', None)
        
        # Check if already logged in
        try:
            # Try to initialize WandB
            self.run = self.wandb.init(
                project=self.project_name,
                name=self.run_name,
                entity=entity,
                tags=tags,
                notes=notes,
                config=config,
                resume='allow'
            )
            
            print(f"✓ WandB plugin initialized")
            print(f"  Project: {self.project_name}")
            if self.run_name:
                print(f"  Run: {self.run_name}")
            print(f"  Dashboard: {self.run.get_url()}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  WandB plugin: Failed to initialize")
            print(f"   Error: {e}")
            print(f"   Make sure you're logged in: wandb login")
            return False
    
    def on_train_begin(self, trainer) -> None:
        """Called when training begins"""
        if not self.run:
            return
        
        try:
            # Log model architecture
            if hasattr(trainer.model, 'count_parameters'):
                self.wandb.log({
                    'model/parameters': trainer.model.count_parameters()
                })
            
            # Watch model (logs gradients and parameters)
            self.wandb.watch(trainer.model, log='all', log_freq=100)
            
            print(f"[{self.name}] Started tracking training")
            
        except Exception as e:
            print(f"⚠️  WandB plugin error in on_train_begin: {e}")
    
    def on_step_end(self, trainer, step: int, metrics: Dict[str, Any]) -> None:
        """
        Called after each training step
        
        Args:
            trainer: Trainer instance
            step: Current step
            metrics: Step metrics
        """
        if not self.run:
            return
        
        # Log at specified interval
        if step % self.log_interval != 0:
            return
        
        try:
            # Prepare metrics for logging
            log_dict = {
                'train/step': step,
            }
            
            # Add all metrics with proper prefixes
            for key, value in metrics.items():
                if key == 'loss':
                    log_dict['train/loss'] = value
                elif key == 'val_loss':
                    log_dict['val/loss'] = value
                elif key == 'lr' or key == 'learning_rate':
                    log_dict['train/learning_rate'] = value
                elif key == 'tokens_per_sec':
                    log_dict['performance/tokens_per_sec'] = value
                elif key == 'gpu_memory_gb':
                    log_dict['system/gpu_memory_gb'] = value
                elif key == 'epoch':
                    log_dict['train/epoch'] = value
                else:
                    log_dict[f'metrics/{key}'] = value
            
            # Log to WandB
            self.wandb.log(log_dict, step=step)
            
        except Exception as e:
            print(f"⚠️  WandB plugin error in on_step_end: {e}")
    
    def on_validation_end(self, trainer, metrics: Dict[str, Any]) -> None:
        """
        Called when validation ends
        
        Args:
            trainer: Trainer instance
            metrics: Validation metrics
        """
        if not self.run:
            return
        
        try:
            # Log validation metrics
            log_dict = {}
            
            for key, value in metrics.items():
                if key == 'loss':
                    log_dict['val/loss'] = value
                elif key == 'perplexity':
                    log_dict['val/perplexity'] = value
                elif key == 'accuracy':
                    log_dict['val/accuracy'] = value
                else:
                    log_dict[f'val/{key}'] = value
            
            if log_dict:
                self.wandb.log(log_dict)
            
        except Exception as e:
            print(f"⚠️  WandB plugin error in on_validation_end: {e}")
    
    def on_checkpoint_save(self, trainer, checkpoint_path: str) -> None:
        """
        Called when a checkpoint is saved
        
        Args:
            trainer: Trainer instance
            checkpoint_path: Path to saved checkpoint
        """
        if not self.run:
            return
        
        try:
            # Log checkpoint as artifact
            artifact = self.wandb.Artifact(
                name=f'model-checkpoint',
                type='model',
                description=f'Model checkpoint at step {trainer.global_step}'
            )
            
            artifact.add_file(checkpoint_path)
            self.run.log_artifact(artifact)
            
            print(f"[{self.name}] Logged checkpoint artifact: {checkpoint_path}")
            
        except Exception as e:
            print(f"⚠️  WandB plugin error in on_checkpoint_save: {e}")
    
    def on_train_end(self, trainer, final_metrics: Dict[str, Any]) -> None:
        """
        Called when training ends
        
        Args:
            trainer: Trainer instance
            final_metrics: Final training metrics
        """
        if not self.run:
            return
        
        try:
            # Log final metrics
            log_dict = {}
            for key, value in final_metrics.items():
                log_dict[f'final/{key}'] = value
            
            if log_dict:
                self.wandb.log(log_dict)
            
            # Log summary
            self.wandb.run.summary['training_complete'] = True
            self.wandb.run.summary['final_loss'] = final_metrics.get('loss', None)
            
            print(f"[{self.name}] Training complete. View results at: {self.run.get_url()}")
            
        except Exception as e:
            print(f"⚠️  WandB plugin error in on_train_end: {e}")
    
    def cleanup(self) -> None:
        """Cleanup WandB resources"""
        if self.run:
            try:
                self.wandb.finish()
                print(f"[{self.name}] Finished WandB run")
            except Exception as e:
                print(f"⚠️  WandB plugin error in cleanup: {e}")
