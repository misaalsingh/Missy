import os
import json
from transformers import AutoTokenizer
import Pretraining.data.dataloader as dataloader


# Dataset definitions

GENERAL_DATASETS = {
    "wikipedia": {
        "path": "wikimedia/wikipedia",
        "name": "20231101.en",
        "split": "train",
        "max_gb": 12,
    },
    "openwebtext": {
        "path": "Skylion007/openwebtext",
        "name": None,
        "split": "train",
        "max_gb": 6,
    },
}

INSTRUCTION_DATASETS = {
    "alpaca": {
        "path": "tatsu-lab/alpaca",
        "name": None,
        "split": "train",
        "max_gb": 1,
    },
    "dolly": {
        "path": "databricks/databricks-dolly-15k",
        "name": None,
        "split": "train",
        "max_gb": 1,
    },
}


# Filters and text extraction

def length_filter(example, min_chars=200, max_chars=20_000):
    text = example.get("text") or example.get("content") or example.get("response")
    return isinstance(text, str) and min_chars <= len(text) <= max_chars

def extract_text(example):
    if "text" in example:
        return example["text"]
    if "content" in example:
        return example["content"]
    if "instruction" in example and "output" in example:
        return f"Instruction:\n{example['instruction']}\n\nResponse:\n{example['output']}"
    if "instruction" in example and "response" in example:
        ctx = example.get("context", "")
        return (
            f"Instruction:\n{example['instruction']}\n\n"
            + (f"Context:\n{ctx}\n\n" if ctx else "")
            + f"Response:\n{example['response']}"
        )
    return None


# Batch generator with tokenization 

def batch_generator(dataset, tokenizer, batch_size, max_length, max_bytes):
    batch = []
    total_bytes = 0

    for example in dataset:
        if not length_filter(example):
            continue

        text = extract_text(example)
        if text is None:
            continue

        batch.append(text)

        if len(batch) == batch_size:
            tokenized = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            # compute approximate bytes for input_ids
            batch_size_bytes = sum(len(ids) * 4 for ids in tokenized["input_ids"])
            total_bytes += batch_size_bytes

            if total_bytes >= max_bytes:
                print("⚠️ Dataset GB limit reached, stopping.")
                yield None
                return

            yield tokenized["input_ids"]
            batch = []

    if batch:
        tokenized = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        yield tokenized["input_ids"]


# Main

def main():
    os.makedirs("tokenized_batches", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    loader = dataloader.DataLoader(streaming=True)

    batch_size = 1000
    max_length = 1024

    def process_dataset(name, info):
        dataset = loader.load(info)
        max_bytes = info["max_gb"] * 1024**3
        batch_id = 0
        for tokenized_batch in batch_generator(dataset, tokenizer, batch_size, max_length, max_bytes):
            if tokenized_batch is None:
                break  # GB limit reached

            path = f"tokenized_batches/{name}_batch_{batch_id}.jsonl"
            with open(path, "w") as f:
                for ids in tokenized_batch:
                    f.write(json.dumps(ids) + "\n")
            print(f"Saved {path}")
            batch_id += 1

    print("Processing general datasets...")
    for name, info in GENERAL_DATASETS.items():
        process_dataset(name, info)

    print("\nProcessing instruction datasets...")
    for name, info in INSTRUCTION_DATASETS.items():
        process_dataset(name, info)

if __name__ == "__main__":
    main()
