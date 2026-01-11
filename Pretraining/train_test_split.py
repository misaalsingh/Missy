import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
import re
import random
import glob

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

JSONL_DIR = "data/tokenized_batches"
MEMMAP_DIR = "data/memmap_batches"

PAD_TOKEN_ID = 50256
IGNORE_INDEX = -100

BATCH_SIZE = 8
NUM_WORKERS = 2   
# split of the training data (percentage of open text and instruction set)
split = {'train': .75, 'test': .15, 'val': .1}

class TrainTestSplit:
    
    def __init__(self, base_dir, split):
        self.base_dir = base_dir
        self.split = split  # e.g., {'train': 0.8, 'test': 0.1, 'val': 0.1}

    def split_path(self):
        l_files = os.listdir(self.base_dir)

        # regex patterns
        pattern_alpaca = re.compile(r"^alpaca_batch_\d+\.jsonl$")
        pattern_dolly  = re.compile(r"^dolly_batch_\d+\.jsonl$")

        # instruction data
        inst_train = [f for f in l_files if pattern_alpaca.match(f)]
        inst_val   = [f for f in l_files if pattern_dolly.match(f)]

        # general data (everything else)
        gen = [f for f in l_files if f not in inst_train + inst_val]
        random.shuffle(gen)

        n = len(gen)
        n_train = int(self.split['train'] * n)
        n_test  = int(self.split['test'] * n)
        n_val   = n - n_train - n_test

        training_files = gen[:n_train]
        test_files     = gen[n_train:n_train+n_test]
        val_files      = gen[n_train+n_test:]

        # add instruction files to appropriate splits
        training_files.extend(inst_train)
        val_files.extend(inst_val)

        return training_files, test_files, val_files

def jsonl_to_memmap(jsonl_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    sequences = []
    with open(jsonl_path, "r") as f:
        for line in f:
            sequences.append(json.loads(line))

    arr = np.asarray(sequences, dtype=np.int32)

    out_path = os.path.join(
        output_dir,
        os.path.basename(jsonl_path).replace(".jsonl", ".npy")
    )

    np.save(out_path, arr)
    print(f"Saved memmap: {out_path}")


def convert_all_jsonl(jsonl_files: List[str], output_dir: str):
    for f in jsonl_files:
        jsonl_to_memmap(f, output_dir)


class MemmapTokenDataset(Dataset):


    def __init__(
        self,
        memmap_files: List[str],
        pad_token_id: int = PAD_TOKEN_ID,
        ignore_index: int = IGNORE_INDEX,
    ):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

        self.arrays = []
        self.index_map = []

        for file_idx, path in enumerate(memmap_files):
            arr = np.load(path, mmap_mode="r")
            self.arrays.append(arr)

            for local_idx in range(arr.shape[0]):
                self.index_map.append((file_idx, local_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, local_idx = self.index_map[idx]
        input_ids = self.arrays[file_idx][local_idx]

        attention_mask = (input_ids != self.pad_token_id).astype(np.int64)

        labels = input_ids.copy()
        labels[input_ids == self.pad_token_id] = self.ignore_index

        return {
            "input_ids": torch.from_numpy(input_ids).long(),
            "attention_mask": torch.from_numpy(attention_mask).long(),
            "labels": torch.from_numpy(labels).long(),
        }


def build_dataloader(memmap_files):
    dataset = MemmapTokenDataset(memmap_files)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,  # MPS-safe
    )
    return loader

if __name__ == "__main__":
    # splitter = TrainTestSplit(JSONL_DIR, split)
    # train_jsonl, test_jsonl, val_jsonl = splitter.split_path()

    # # Convert JSONL → memmap (only once per split)
    # for split_name, files in {
    #     "train": train_jsonl,
    #     "test": test_jsonl,
    #     "val": val_jsonl,
    # }.items():
    #     out_dir = os.path.join(MEMMAP_DIR, split_name)
    #     if not os.path.exists(out_dir) or not os.listdir(out_dir):
    #         print(f"Converting {split_name} JSONL files...")
    #         convert_all_jsonl(
    #             [os.path.join(JSONL_DIR, f) for f in files],
    #             out_dir
    #         )

    # Build loaders
    train_loader = build_dataloader(
        glob.glob(os.path.join(MEMMAP_DIR, "train", "*.npy"))
    )

    val_loader = build_dataloader(
        glob.glob(os.path.join(MEMMAP_DIR, "val", "*.npy"))
    )

    test_loader = build_dataloader(
        glob.glob(os.path.join(MEMMAP_DIR, "test", "*.npy"))
    )

    # Sanity check
    batch = next(iter(train_loader))
    print("Batch keys:", batch.keys())
    print("input_ids shape:", batch["input_ids"].shape)


