from utils.config import Config
import yaml
from dataloaders.BaseDataWrapper import BaseDataWrapper
import torch

class ShuffleLoader(BaseDataWrapper):
    """
    This class is used to shuffle the samples from multiple dataloaders.
    """
    def __init__(self, data_cfg, **kwargs):
        
        self.data_cfg = data_cfg
        self.batch_size = data_cfg.batch_size
        from dataloaders import setup_dataloader
        self.batches = []
        self.loaders = []
        for path in data_cfg.loaders:
            with open(path, 'r') as file:
                data = yaml.safe_load(file)
            _data_cfg = Config.from_dict(data)
            _data_cfg = _data_cfg.TrainConfig.DataConfig
            for key, value in data_cfg.items():
                if key in ["loader", "dataset_type"]:
                    continue
                if key in _data_cfg:
                    setattr(_data_cfg, key, value)
            self.loaders.append(setup_dataloader(data_cfg=_data_cfg))

    def __iter__(self):
        self.iterators = [iter(loader) for loader in self.loaders]
        return self

    def __next__(self):
        if len(self.batches) == 0:
            samples = [next(loader) for loader in self.iterators]
            samples = torch.cat(samples, dim=0)
            # Shuffle all samples
            samples = samples[torch.randperm(samples.size(0))]
            # Return the shuffled tensor
            batches = torch.split(samples, self.batch_size, dim=0)
            self.batches = list(batches)
        return self.batches.pop()
