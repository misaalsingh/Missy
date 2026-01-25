# data/train_test_split.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob

PAD_TOKEN_ID = 50256
IGNORE_INDEX = -100


class MemmapTokenDataset(Dataset):
    """Clean, simple dataset for memmap files"""
    
    def __init__(self, memmap_files, pad_token_id=PAD_TOKEN_ID, ignore_index=IGNORE_INDEX):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        
        # Load all data into memory (or use mmap if too large)
        self.data = []
        for path in memmap_files:
            arr = np.load(path, mmap_mode='r')  # Memory-mapped for efficiency
            self.data.append(arr)
        
        # Concatenate if small enough, otherwise keep separate
        if len(self.data) == 1:
            self.data = self.data[0]
        else:
            # For multiple files, build an index
            self.index_map = []
            for file_idx, arr in enumerate(self.data):
                for seq_idx in range(len(arr)):
                    self.index_map.append((file_idx, seq_idx))
    
    def __len__(self):
        if isinstance(self.data, np.ndarray):
            return len(self.data)
        return len(self.index_map)
    
    def __getitem__(self, idx):
        # Get sequence
        if isinstance(self.data, np.ndarray):
            input_ids = self.data[idx]
        else:
            file_idx, seq_idx = self.index_map[idx]
            input_ids = self.data[file_idx][seq_idx]
        
        # Create labels (same as input_ids for causal LM)
        # Your model will handle the shift internally
        labels = input_ids.copy()
        labels[input_ids == self.pad_token_id] = self.ignore_index
        
        return {
            "input_ids": torch.from_numpy(input_ids.copy()).long(),
            "labels": torch.from_numpy(labels.copy()).long(),
        }


def build_dataloader(memmap_files, batch_size=12, shuffle=True, num_workers=0):
    """Build DataLoader from memmap files"""
    dataset = MemmapTokenDataset(memmap_files)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # MPS doesn't support pin_memory
    )
    
    return loader


if __name__ == "__main__":
    # Test the dataset
    import sys
    
    data_dir = "data/memmap_batches"
    
    for split in ['train', 'val', 'test']:
        files = glob.glob(os.path.join(data_dir, split, "*.npy"))
        if not files:
            print(f"❌ No files found for {split}")
            continue
        
        loader = build_dataloader(files, batch_size=8)
        batch = next(iter(loader))
        
        print(f"\n{split.upper()} split:")
        print(f"  Batch shape: {batch['input_ids'].shape}")
        print(f"  First 20 tokens: {batch['input_ids'][0, :20].tolist()}")
        print(f"  Labels first 20: {batch['labels'][0, :20].tolist()}")
        print(f"  Padding ratio: {(batch['input_ids'] == PAD_TOKEN_ID).float().mean():.1%}")