"""
Logging callback for tracking training metrics
"""

import time
from typing import Any, Dict, Optional
from pathlib import Path
from .base import Callback


class LoggingCallback(Callback):
    """
    Logs training metrics to console, file, and TensorBoard
    
    Features:
    - Log at regular intervals
    - Track loss, learning rate, tokens/sec
    - Save logs to file
    - TensorBoard integration
    - Display progress
    """
    
    def __init__(
        self,
        log_interval: int = 100,
        log_dir: str = 'logs',
        log_file: str = 'training.log',
        verbose: bool = True,
        use_tensorboard: bool = True
    ):
        """
        Initialize logging callback
        
        Args:
            log_interval: Log every N steps
            log_dir: Directory for log files
            log_file: Log file name
            verbose: Print to console
            use_tensorboard: Enable TensorBoard logging
        """
        self.log_interval = log_interval
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        self.verbose = verbose
        self.use_tensorboard = use_tensorboard
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = None
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tensorboard_dir = self.log_dir / 'tensorboard'
                self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
                if self.verbose:
                    print(f"TensorBoard logging enabled: {tensorboard_dir}")
                    print(f"View with: tensorboard --logdir={tensorboard_dir}")
            except ImportError:
                if self.verbose:
                    print("⚠️  TensorBoard not available. Install with: pip install tensorboard")
                self.writer = None
        
        self.start_time = None
        self.step_times = []
        self.losses = []
    
    def on_train_begin(self, trainer: Any):
        """Initialize logging"""
        self.start_time = time.time()
        
        log_path = self.log_dir / self.log_file
        with open(log_path, 'w') as f:
            f.write("step,loss,lr,tokens_per_sec,elapsed_time\n")
        
        if self.verbose:
            print(f"Training logs will be saved to: {log_path}")
            print(f"{'='*60}")
            print(f"{'Step':<10} {'Loss':<12} {'LR':<12} {'Tokens/s':<12} {'Time'}")
            print(f"{'='*60}")
    
    def on_step_end(self, trainer: Any, step: int, loss: float, metrics: Optional[Dict[str, float]] = None):
        """Log metrics at intervals"""
        self.losses.append(loss)
        
        if step % self.log_interval == 0 and step > 0:
            self._log_metrics(trainer, step, loss, metrics)
    
    def on_validation_end(self, trainer: Any, step: int, val_loss: float, val_metrics: Optional[Dict[str, float]] = None):
        """Log validation metrics to TensorBoard"""
        if self.writer is not None:
            self.writer.add_scalar('val/loss', val_loss, step)
            if val_metrics:
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{key}', value, step)
    
    def on_train_end(self, trainer: Any):
        """Log final metrics and close TensorBoard writer"""
        elapsed = time.time() - self.start_time
        avg_loss = sum(self.losses) / len(self.losses) if self.losses else 0
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
        
        if self.verbose:
            print(f"{'='*60}")
            print(f"Training completed!")
            print(f"Total time: {self._format_time(elapsed)}")
            print(f"Average loss: {avg_loss:.4f}")
            print(f"{'='*60}")
    
    def _log_metrics(self, trainer: Any, step: int, loss: float, metrics: Optional[dict]):
        """Log metrics to console, file, and TensorBoard"""
        elapsed = time.time() - self.start_time
        
        # Calculate tokens per second
        tokens_per_sec = 0
        if hasattr(trainer, 'tokens_processed'):
            tokens_per_sec = trainer.tokens_processed / elapsed
        
        # Get learning rate
        lr = trainer.optimizer.param_groups[0]['lr']
        
        # Log to file
        log_path = self.log_dir / self.log_file
        with open(log_path, 'a') as f:
            f.write(f"{step},{loss:.6f},{lr:.6e},{tokens_per_sec:.2f},{elapsed:.2f}\n")
        
        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('train/loss', loss, step)
            self.writer.add_scalar('train/learning_rate', lr, step)
            self.writer.add_scalar('train/tokens_per_sec', tokens_per_sec, step)
            if metrics:
                for key, value in metrics.items():
                    self.writer.add_scalar(f'train/{key}', value, step)
        
        # Log to console
        if self.verbose:
            time_str = self._format_time(elapsed)
            print(f"{step:<10} {loss:<12.4f} {lr:<12.6e} {tokens_per_sec:<12.0f} {time_str}")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
