from datasets import load_dataset

class DataLoader:
    def __init__(
        self,
        streaming: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.streaming = streaming
        self.rank = rank
        self.world_size = world_size

    def load(self, dataset_info: dict):
        """
        Loads and shards a Hugging Face dataset.
        """
        dataset = load_dataset(
            dataset_info["path"],
            dataset_info.get("name"),
            split=dataset_info["split"],
            streaming=self.streaming,
            trust_remote_code=True,
        )

        # Shard if running multi-process / multi-GPU
        if self.world_size > 1:
            dataset = dataset.shard(
                num_shards=self.world_size,
                index=self.rank,
                contiguous=True,
            )

        return dataset
