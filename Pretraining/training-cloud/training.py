import os
import time
import math
import pickle
import json
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import GPTConfig, GPT
from data.train_test_split import build_dataloader
import glob
from datetime import datetime
from logger import TrainingLogger
import wandb

# -----------------------------------------------------------------------------

out_dir = 'out'
eval_interval = 50
log_interval = 1
eval_iters = 200
eval_only = False 
always_save_checkpoint = True 
init_from = 'scratch' 
wandb_log = False 
wandb_project = 'owt'
wandb_run_name = 'gpt2' 


# data

gradient_accumulation_steps = 5 * 8 
batch_size = 12 
block_size = 1024


# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 
bias = False 

max_iters = 600000 
beta1 = 0.9
beta2 = 0.95

learning_rate = 3e-4   
min_lr = 3e-5             
grad_clip = 0.5
weight_decay = 1e-2
warmup_iters = 500

decay_lr = True 
lr_decay_iters = 600000 

backend = 'nccl' 
# system
device = 'mps' #switch to cuda
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False 
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) 
config = {k: globals()[k] for k in config_keys} 


ddp = int(os.environ.get('RANK', -1)) != -1 
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'mps:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 
    seed_offset = ddp_rank 
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
device_type = device if device in device else 'cpu' 


ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# Fixed version - only use autocast for CUDA
if device_type == 'cuda':
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
else:
    ctx = nullcontext()  


data_dir = 'data/memmap_batches'

train_loader = build_dataloader(
    glob.glob(os.path.join(data_dir, "train", "*.npy"))
)

val_loader = build_dataloader(
    glob.glob(os.path.join(data_dir, "val", "*.npy"))
)

train_iter = None
val_iter = None
def get_batch(split):
    global train_iter, val_iter

    if split == "train":
        loader = train_loader
        if train_iter is None:
            train_iter = iter(loader)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)

    elif split == "val":
        loader = val_loader
        if val_iter is None:
            val_iter = iter(loader)
        try:
            batch = next(val_iter)
        except StopIteration:
            val_iter = iter(loader)
            batch = next(val_iter)

    else:
        raise ValueError(f"Unknown split: {split}")

    x = batch["input_ids"].to(device)
    y = batch["labels"].to(device)
    return x, y


iter_num = 0
best_val_loss = 1e9
vocab_size = 50257


# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    
    print("Initializing a new model from scratch")
    if vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = vocab_size if vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
   
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']


    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")


    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)

    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)


if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size 
model.to(device)


scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16')) if dtype == 'float16' else torch.amp.GradScaler('cpu', enabled=False)


optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None 


if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) 


if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):

    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)

    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (learning_rate - min_lr)

# logging

logger = TrainingLogger()
logger.log_config(config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
print(X.shape, Y.shape)
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        
        # Enhanced logging for evaluation
        logger.log_evaluation(iter_num, losses['train'], losses['val'], lr)
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, 
            })
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                torch.save(checkpoint, ckpt_path)
                logger.log_checkpoint(iter_num, ckpt_path)
    
    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps 
        
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: 
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        # Enhanced logging for iteration
        logger.log_iteration(iter_num, lossf, dt, running_mfu, lr)
    
    iter_num += 1
    local_iter_num += 1

    # check termination conditions
    if iter_num > max_iters:
        break

# Training complete
logger.log_summary(iter_num, lossf)

# Create completion marker for auto-shutdown (cloud only)
if master_process:
    try:
        with open('/home/training/TRAINING_COMPLETE', 'w') as f:
            f.write(f'Training completed at iteration {iter_num}\n')
            f.write(f'Final loss: {lossf:.4f}\n')
            f.write(f'Best val loss: {best_val_loss:.4f}\n')
            f.write(f'Total time: {(time.time() - logger.start_time)/3600:.2f} hours\n')
    except:
        pass  # File path doesn't exist on local, that's fine

if ddp:
    destroy_process_group()