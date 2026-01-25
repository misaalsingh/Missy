import json
import matplotlib.pyplot as plt
import glob
import os

def plot_training_metrics(metrics_file):
    
    # Load metrics
    train_iters, train_losses, train_lrs = [], [], []
    eval_iters, eval_train_losses, eval_val_losses = [], [], []
    
    with open(metrics_file, 'r') as f:
        for line in f:
            m = json.loads(line)
            
            if m['type'] == 'train':
                train_iters.append(m['iter'])
                train_losses.append(m['loss'])
                train_lrs.append(m['lr'])
            
            elif m['type'] == 'eval':
                eval_iters.append(m['iter'])
                eval_train_losses.append(m['train_loss'])
                eval_val_losses.append(m['val_loss'])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    axes[0, 0].plot(train_iters, train_losses, alpha=0.6, label='Train (per iter)')
    axes[0, 0].plot(eval_iters, eval_train_losses, 'o-', label='Train (eval)', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Val loss
    axes[0, 1].plot(eval_iters, eval_val_losses, 'o-', color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Train vs Val
    axes[1, 0].plot(eval_iters, eval_train_losses, 'o-', label='Train', linewidth=2)
    axes[1, 0].plot(eval_iters, eval_val_losses, 'o-', label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Train vs Val Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 1].plot(train_iters, train_lrs, alpha=0.6)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plot_file = metrics_file.replace('.jsonl', '.png')
    plt.savefig(plot_file, dpi=150)
    print(f"Plot saved to {plot_file}")
    plt.show()

if __name__ == "__main__":
    # Find most recent metrics file
    metrics_files = glob.glob('logs/metrics_*.jsonl')
    if metrics_files:
        latest = max(metrics_files, key=os.path.getctime)
        print(f"Plotting: {latest}")
        plot_training_metrics(latest)
    else:
        print("No metrics files found in logs/")