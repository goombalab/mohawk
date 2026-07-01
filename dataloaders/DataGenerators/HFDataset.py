import os

import datasets
import datasets.distributed

from utils.distributed import global_rank, world_size
from utils.logging import logger

datasets.config.STREAMING_READ_MAX_RETRIES = 1_000_000  # default is 20
datasets.config.STREAMING_READ_RETRY_INTERVAL = 5  # default is 5

from dataloaders.BaseDataGenerator import BaseDataGenerator


def _to_builtin(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_builtin(item) for item in value)
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    return value


class HFDataset(BaseDataGenerator):
    def __init__(
        self,
        path: str,
        name: str = None,
        streaming: bool = False,
        split: str = "train",
        skip_samples: int = 0,
        shuffle_buffer_size: int = 10000,
        **dataset_kwargs,
    ):
        # setup_dataloader threads the previous pipeline stage through every
        # constructor; source datasets have no upstream iterable.
        dataset_kwargs.pop("iterable", None)
        dataset_kwargs = {
            key: _to_builtin(value) for key, value in dataset_kwargs.items()
        }
        self._initialize(
            path=path,
            name=name,
            streaming=streaming,
            split=split,
            skip_samples=skip_samples,
            shuffle_buffer_size=shuffle_buffer_size,
            dataset_kwargs=dataset_kwargs,
        )

    def _initialize(
            self,
            path: str,
            name: str = None,
            streaming: bool = False,
            split: str = "train",
            skip_samples: int = 0,
            _index: int = 0,
            shuffle_buffer_size: int = 10000,
            dataset_kwargs: dict = None,
        ):
        """
        Initialize the dataset.
        """
        self.path = path
        self.name = name
        self.streaming = streaming
        self.split = split
        self.shuffle_buffer_size = shuffle_buffer_size
        self.dataset_kwargs = dict(dataset_kwargs or {})
        self._index = max(0, _index)
        self.in_use = False
        self._iterator = None

        # Initialize the dataset
        self.dataset = datasets.load_dataset(
            path=self.path,
            name=self.name,
            streaming=self.streaming,
            cache_dir=os.environ.get("HF_DATASETS_CACHE"),
            **self.dataset_kwargs,
            )
        
        assert self.split in self.dataset.keys(), f"[HFDataset] Split {self.split} not found in dataset."
        self.dataset = self.dataset[self.split]

        if self.streaming:
            self.dataset = self.dataset.shuffle(seed=42, buffer_size=self.shuffle_buffer_size)

        if self._index > 0 and skip_samples > 0:
            logger.warning(f"[HFDataset] Cannot skip samples and set index at the same time. Skipping samples will be ignored.")
            skip_samples = 0
            
        self._index += skip_samples

        if self._index > 0:
            self.dataset = self.dataset.skip(self._index)
            logger.info(f"[HFDataset] Starting {self.path} from index {self._index}.")

        self.dataset = datasets.distributed.split_dataset_by_node(
            dataset=self.dataset, 
            rank=global_rank, 
            world_size=world_size
            ) if world_size > 1 else self.dataset
        
        # WARNING: from now on, no skip_samples are allowed, as the dataset is already split

    def __str__(self, tabs=0, **kwargs):
        return f"---"*tabs + f"HFDataset (path={self.path}, name={self.name}, split={self.split}, streaming={self.streaming})"
        
    def state_dict(self):
        """
        Returns the state of the dataset.

        Note: no need to save skip_samples, as it is handled by _index.
        """
        return {
            "path": self.path,
            "name": self.name,
            "streaming": self.streaming,
            "split": self.split,
            "shuffle_buffer_size": self.shuffle_buffer_size,
            "dataset_kwargs": self.dataset_kwargs,
            "_index": self._index
        }
    
    def load_state_dict(self, state_dict):
        assert self.path == state_dict["path"], f"[HFDataset] Cannot load state dict from a different dataset. Current path: {self.path}, state dict path: {state_dict['path']}"
        assert self.name == state_dict["name"], f"[HFDataset] Cannot load state dict from a different dataset. Current name: {self.name}, state dict name: {state_dict['name']}"
        assert self.split == state_dict["split"], f"[HFDataset] Cannot load state dict from a different dataset. Current split: {self.split}, state dict split: {state_dict['split']}"
        self._initialize(**state_dict)

    def __iter__(self):
        if self.in_use:
            raise StopIteration("Dataset has already been iterated over. Please create a new instance of the dataset.")

        self.in_use = True
        self._iterator = iter(self.dataset)

        try:
            for sample in self._iterator:
                self._index += 1
                yield sample
        finally:
            self.close()
            self.in_use = False

    def close(self):
        iterator = self._iterator
        self._iterator = None
        if iterator is None:
            return

        close = getattr(iterator, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception as exc:
            logger.warning(f"[HFDataset] Failed to close streaming iterator cleanly: {exc}")
