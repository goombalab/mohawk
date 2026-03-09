from dataloaders.BaseDataGenerator import BaseDataGenerator
from dataloaders.DataWrappers.TokenizedDataLoader import TokenizedDataLoader
import torch
from utils.logging import logger

class SequentialLoader(BaseDataGenerator, TokenizedDataLoader):
    """
    A data generator that iterates over multiple dataloaders in a sequential fashion.
    It switches to the next dataloader only when the current one is exhausted. 
    It processes batches from each dataloader in turn, allowing each dataloader to complete 
    its available batches before moving on to the next.
    """
    def __init__(self, train_data_configs, max_samples=torch.inf, **kwargs):
        from dataloaders import setup_dataloader
        self.iterators = None
        self.max_samples = max_samples
        self.train_data_configs = train_data_configs
        self.samples_seen = 0
        self._idx = 0
        self.loaders = [setup_dataloader(config.TrainDataConfig) for config in train_data_configs]

        # assert all dataloaders have the same tokenizer
        self._tokenizer = self.loaders[0].tokenizer
        assert all([self._tokenizer.name_or_path == loader.tokenizer.name_or_path for loader in self.loaders])

    def __iter__(self):
        self.iterators = [iter(loader) for loader in self.loaders]
        return self
    
    def __next__(self):
        try:
            if self.samples_seen >= self.max_samples:
                raise StopIteration
            next_batch = next(self.iterators[self._idx])
        except StopIteration:
            self.samples_seen = 0
            self._idx = (self._idx + 1) % len(self.loaders)
            next_batch = next(self.iterators[self._idx])
        
        if self.samples_seen == 0:
            logger.info(f"[SequentialLoader] Setting dataloader to {self._idx}")
        
        self.samples_seen += 1
        return next_batch
