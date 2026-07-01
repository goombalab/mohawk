from torch.utils.data import DataLoader, IterableDataset
from dataloaders.BaseDataWrapper import BaseDataWrapper
from utils.logging import logger
import torch

class TorchDataLoader(DataLoader, BaseDataWrapper):
    def __init__(
        self,
        iterable,
        num_workers=0,
        prefetch_factor=None,
        batch_size=1,
        **kwargs
    ):
        if isinstance(iterable, IterableDataset) and num_workers > 0:
            logger.warning(f"[WARNING] You are using num_workers > 0 with IterableDataset."
                           f"This may cause duplicate data loading.")

        super().__init__(
            iterable, 
            num_workers=num_workers, 
            prefetch_factor=prefetch_factor, 
            batch_size=batch_size
            )
        # call BaseDataWrapper super class
        BaseDataWrapper.__init__(self, iterable, **kwargs)

    def __iter__(self):
        iterator = super().__iter__()
        try:
            for batch in iterator:
                if isinstance(batch, torch.Tensor):
                    yield {"input_ids": batch}
                else:
                    yield batch
        finally:
            fetcher = getattr(iterator, "_dataset_fetcher", None)
            dataset_iter = getattr(fetcher, "dataset_iter", None)
            close_dataset_iter = getattr(dataset_iter, "close", None)
            if close_dataset_iter is not None:
                close_dataset_iter()
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
            shutdown_workers = getattr(iterator, "_shutdown_workers", None)
            if shutdown_workers is not None:
                shutdown_workers()
