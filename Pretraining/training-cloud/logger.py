import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import GPTConfig, GPT
from data.train_test_split import build_dataloader
import glob
import json
from datetime import datetime

# -----------------------------------------------------------------------------
# ENHANCED LOGGING SETUP
# -----------------------------------------------------------------------------

class TrainingLogger:
    """Enhanced logging for training metrics"""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        self.metrics_file = os.path.join(log_dir, f'metrics_{timestamp}.jsonl')
        
        # Initialize
        self.start_time = time.time()
        self.best_val_loss = float('inf')
        
        self.log("=" * 80)
        self.log(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 80)
    
    def log(self, message):
        """Log message to both console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        with open(self.log_file, 'a') as f:
            f.write(log_line + '\n')
    
    def log_metrics(self, iter_num, metrics):
        """Log metrics as JSON for easy plotting later"""
        metrics['iter'] = iter_num
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['elapsed_hours'] = (time.time() - self.start_time) / 3600
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def log_iteration(self, iter_num, loss, dt, mfu, lr):
        """Log training iteration"""
        self.log(f"iter {iter_num}: loss {loss:.4f}, time {dt*1000:.2f}ms, mfu {mfu*100:.2f}%, lr {lr:.2e}")
        
        self.log_metrics(iter_num, {
            'type': 'train',
            'loss': loss,
            'time_ms': dt * 1000,
            'mfu': mfu,
            'lr': lr
        })
    
    def log_evaluation(self, iter_num, train_loss, val_loss, lr):
        """Log evaluation results"""
        # Convert tensors to float
        train_loss = float(train_loss)
        val_loss = float(val_loss)
        
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        
        self.log("-" * 80)
        self.log(f"EVAL @ iter {iter_num}:")
        self.log(f"  Train loss: {train_loss:.4f}")
        self.log(f"  Val loss:   {val_loss:.4f} {'✨ NEW BEST!' if is_best else ''}")
        self.log(f"  Best val:   {self.best_val_loss:.4f}")
        self.log(f"  LR:         {lr:.2e}")
        self.log("-" * 80)
        
        self.log_metrics(iter_num, {
            'type': 'eval',
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'is_best': is_best,
            'lr': float(lr)
        })
        
    def log_checkpoint(self, iter_num, path):
        """Log checkpoint save"""
        self.log(f"💾 Checkpoint saved: {path} (iter {iter_num})")
    
    def log_config(self, config):
        """Log training configuration"""
        self.log("\n" + "=" * 80)
        self.log("TRAINING CONFIGURATION:")
        self.log("=" * 80)
        for key, value in sorted(config.items()):
            self.log(f"  {key:30s} = {value}")
        self.log("=" * 80 + "\n")
    
    def log_summary(self, total_iters, final_loss):
        """Log training summary"""
        elapsed = time.time() - self.start_time
        self.log("\n" + "=" * 80)
        self.log("TRAINING COMPLETE!")
        self.log("=" * 80)
        self.log(f"  Total iterations: {total_iters}")
        self.log(f"  Final loss:       {final_loss:.4f}")
        self.log(f"  Best val loss:    {self.best_val_loss:.4f}")
        self.log(f"  Total time:       {elapsed/3600:.2f} hours")
        self.log(f"  Avg time/iter:    {elapsed/total_iters:.2f} seconds")
        self.log("=" * 80)
