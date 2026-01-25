import os
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import random

# Configuration
MEMMAP_DIR = "memmap_batches"
TOKENIZER_NAME = "gpt2"
MAX_LENGTH = 256
BATCH_SIZE = 1000  # Documents to tokenize at once

# Dataset definitions
DATASETS = {
    "openwebtext": {
        "path": "Skylion007/openwebtext",
        "name": None,
        "split": "train",
        "max_gb": 6,
        "text_key": "text",
    },
    "wikipedia": {
        "path": "wikimedia/wikipedia",
        "name": "20231101.en",
        "split": "train",
        "max_gb": 12,
        "text_key": "text",
    },
}

# Train/val/test split ratios
SPLIT_RATIOS = {'train': 0.75, 'val': 0.15, 'test': 0.10}

PAD_TOKEN_ID = 50256


def length_filter(text, min_chars=200, max_chars=20_000):
    return isinstance(text, str) and min_chars <= len(text) <= max_chars


def tokenize_and_save_batches(dataset_name, dataset_config, tokenizer):

    print(f"\n Processing {dataset_name}...")
    
    dataset = load_dataset(
        dataset_config["path"],
        dataset_config.get("name"),
        split=dataset_config["split"],
        streaming=True,
        trust_remote_code=True,
    )
    
    text_key = dataset_config.get("text_key", "text")
    max_bytes = dataset_config["max_gb"] * 1024**3
    total_bytes = 0
    batch_files = []
    
    batch_texts = []
    batch_id = 0
    
    for example in tqdm(dataset, desc=f"Tokenizing {dataset_name}"):
        # Extract text
        text = example.get(text_key)
        if not text or not length_filter(text):
            continue
        
        batch_texts.append(text)
        

        if len(batch_texts) >= BATCH_SIZE:
            tokenized = tokenizer(
                batch_texts,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
                return_tensors="np",
            )
            
            # Save as numpy array
            arr = tokenized["input_ids"].astype(np.int32)
            batch_path = f"temp_{dataset_name}_batch_{batch_id}.npy"
            np.save(batch_path, arr)
            batch_files.append(batch_path)
            

            batch_size_bytes = arr.nbytes
            total_bytes += batch_size_bytes
            
            print(f"  Saved batch {batch_id}: {arr.shape[0]} sequences, "
                  f"{total_bytes / 1024**3:.2f}GB total")
            
            batch_texts = []
            batch_id += 1
            

            if total_bytes >= max_bytes:
                print(f"  ✅ Reached {dataset_config['max_gb']}GB limit for {dataset_name}")
                break
    

    if batch_texts:
        tokenized = tokenizer(
            batch_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="np",
        )
        arr = tokenized["input_ids"].astype(np.int32)
        batch_path = f"temp_{dataset_name}_batch_{batch_id}.npy"
        np.save(batch_path, arr)
        batch_files.append(batch_path)
        print(f"  Saved final batch {batch_id}: {arr.shape[0]} sequences")
    
    print(f"✅ {dataset_name}: {len(batch_files)} batches, "
          f"{total_bytes / 1024**3:.2f}GB total")
    
    return batch_files


def split_and_organize_batches(all_batch_files):

    print("\n Shuffling and splitting data...")
    
    all_sequences = []
    for batch_file in tqdm(all_batch_files, desc="Loading batches"):
        arr = np.load(batch_file)
        all_sequences.append(arr)
    
    all_data = np.concatenate(all_sequences, axis=0)
    print(f"Total sequences: {all_data.shape[0]:,}")
    
    np.random.shuffle(all_data)
    
    n_total = len(all_data)
    n_train = int(SPLIT_RATIOS['train'] * n_total)
    n_val = int(SPLIT_RATIOS['val'] * n_total)
    
    splits = {
        'train': all_data[:n_train],
        'val': all_data[n_train:n_train + n_val],
        'test': all_data[n_train + n_val:],
    }
    
    for split_name, split_data in splits.items():
        split_dir = os.path.join(MEMMAP_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        output_path = os.path.join(split_dir, f"{split_name}_data.npy")
        np.save(output_path, split_data)
        
        print(f"  {split_name}: {split_data.shape[0]:,} sequences "
              f"({split_data.nbytes / 1024**3:.2f}GB) → {output_path}")

    print("\n🧹 Cleaning up temporary files...")
    for batch_file in all_batch_files:
        os.remove(batch_file)
    
    print("✅ Data preparation complete!")

def verify_data():
    
    print("\n🔍 Verifying data...")
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(MEMMAP_DIR, split)
        files = [f for f in os.listdir(split_dir) if f.endswith('.npy')]
        
        if files:
            data = np.load(os.path.join(split_dir, files[0]))
            print(f"\n{split.upper()}:")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  First sequence (first 20 tokens): {data[0, :20]}")
            print(f"  Unique tokens in first seq: {len(np.unique(data[0]))}")
            print(f"  Padding ratio: {(data == PAD_TOKEN_ID).mean():.1%}")


def main():
    print(" Starting data preparation pipeline...")
    
    print(f"\n📝 Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    all_batch_files = []
    for dataset_name, dataset_config in DATASETS.items():
        batch_files = tokenize_and_save_batches(
            dataset_name, 
            dataset_config, 
            tokenizer
        )
        all_batch_files.extend(batch_files)
    
    split_and_organize_batches(all_batch_files)
    
    verify_data()
    
    print("\n✅ All done! Data ready for training.")
    print(f"📂 Data location: {MEMMAP_DIR}")


if __name__ == "__main__":
    
    random.seed(42)
    np.random.seed(42)
    
    main()