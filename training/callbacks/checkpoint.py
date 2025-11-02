"""
Checkpoint callback for saving model checkpoints during training
"""

import torch
from pathlib import Path
from typing import Any, Optional
from .base import Callback


class CheckpointCallback(Callback):
    """
    Saves model checkpoints during training
    
    Features:
    - Save at regular intervals
    - Keep only N most recent checkpoints
    - Save best model based on validation loss
    - Save on training interruption
    """
    
    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        save_interval: int = 1000,
        save_total_limit: int = 3,
        save_best: bool = True,
        verbose: bool = True
    ):
        """
        Initialize checkpoint callback
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_interval: Save checkpoint every N steps
            save_total_limit: Maximum number of checkpoints to keep
            save_best: Whether to save best model
            verbose: Print checkpoint messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_interval = save_interval
        self.save_total_limit = save_total_limit
        self.save_best = save_best
        self.verbose = verbose
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_loss = float('inf')
        self.checkpoints = []
    
    def on_train_begin(self, trainer: Any):
        """Setup checkpoint directory"""
        if self.verbose:
            print(f"Checkpoints will be saved to: {self.checkpoint_dir}")
    
    def on_step_end(self, trainer: Any, step: int, loss: float, metrics: Optional[dict] = None):
        """Save checkpoint at intervals"""
        if step % self.save_interval == 0 and step > 0:
            self._save_checkpoint(trainer, step, loss, is_best=False)
    
    def on_train_end(self, trainer: Any):
        """Save final checkpoint"""
        self._save_checkpoint(trainer, trainer.global_step, trainer.last_loss, is_final=True)
    
    def _save_checkpoint(
        self,
        trainer: Any,
        step: int,
        loss: float,
        is_best: bool = False,
        is_final: bool = False
    ):
        """Save a checkpoint"""
        # Determine checkpoint name
        if is_final:
            checkpoint_name = 'checkpoint-final.pt'
        elif is_best:
            checkpoint_name = 'checkpoint-best.pt'
        else:
            checkpoint_name = f'checkpoint-{step}.pt'
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Create checkpoint
        checkpoint = {
            'step': step,
            'epoch': getattr(trainer, 'epoch', 0),
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': loss,
            'config': getattr(trainer, 'config', None),
        }
        
        # Add scheduler if exists
        if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
            checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        if self.verbose:
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Track checkpoint
        if not is_best and not is_final:
            self.checkpoints.append(checkpoint_path)
            self._cleanup_old_checkpoints()
        
        # Save best model
        if self.save_best and loss < self.best_loss:
            self.best_loss = loss
            best_path = self.checkpoint_dir / 'checkpoint-best.pt'
            torch.save(checkpoint, best_path)
            if self.verbose:
                print(f"New best model saved: {best_path} (loss: {loss:.4f})")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to keep only save_total_limit"""
        if len(self.checkpoints) > self.save_total_limit:
            # Remove oldest checkpoint
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                if self.verbose:
                    print(f"Removed old checkpoint: {old_checkpoint}")
    
    def load_checkpoint(self, checkpoint_path: str, trainer: Any):
        """Load a checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and hasattr(trainer, 'scheduler'):
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        trainer.global_step = checkpoint['step']
        trainer.epoch = checkpoint.get('epoch', 0)
        
        if self.verbose:
            print(f"Loaded checkpoint from: {checkpoint_path}")
            print(f"Resuming from step {trainer.global_step}")
        
        return checkpoint
