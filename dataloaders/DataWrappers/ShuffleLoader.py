from utils.config import Config
import yaml
from dataloaders.BaseDataWrapper import BaseDataWrapper
import torch

class ShuffleLoader(BaseDataWrapper):
    """
    This class is used to shuffle the samples from multiple dataloaders.
    """
    def __init__(self, data_cfg, **kwargs):
        kwargs.pop("iterable", None)
        super().__init__(iterable=None, **kwargs)

        self.data_cfg = data_cfg
        self.batch_size = data_cfg.batch_size
        self.max_seq_len = data_cfg.get("max_seq_len", None)
        from dataloaders import setup_dataloader
        self.batches = []
        self.loaders = []
        for path in data_cfg.loaders:
            with open(path, 'r') as file:
                data = yaml.safe_load(file)
            _data_cfg = self._get_train_data_config(Config.from_dict(data), path)
            for key, value in data_cfg.items():
                if key in ["loader", "loaders", "dataset_type"]:
                    continue
                if key in _data_cfg:
                    setattr(_data_cfg, key, value)
            self.loaders.append(setup_dataloader(data_cfg=_data_cfg))

        self._tokenizer = self.loaders[0].tokenizer
        assert all([
            self._tokenizer.name_or_path == loader.tokenizer.name_or_path
            for loader in self.loaders
        ])
        if self.max_seq_len is None:
            self.max_seq_len = self.loaders[0].max_seq_len
            assert all([
                loader.max_seq_len == self.max_seq_len
                for loader in self.loaders
            ])

    @staticmethod
    def _get_train_data_config(config, path):
        if hasattr(config, "TrainDataConfig"):
            return config.TrainDataConfig
        if hasattr(config, "TrainConfig") and hasattr(config.TrainConfig, "DataConfig"):
            return config.TrainConfig.DataConfig
        raise ValueError(
            f"ShuffleLoader source config {path} must define TrainDataConfig "
            "or legacy TrainConfig.DataConfig."
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    def __iter__(self):
        self.iterators = [iter(loader) for loader in self.loaders]
        return self

    def __next__(self):
        if len(self.batches) == 0:
            samples = [next(loader) for loader in self.iterators]
            samples = self._concat_samples(samples)
            samples = self._shuffle_samples(samples)
            self.batches = self._split_samples(samples)
        return self.batches.pop()

    @staticmethod
    def _ensure_batch_dim(tensor):
        return tensor.unsqueeze(0) if tensor.dim() == 1 else tensor

    def _concat_samples(self, samples):
        first = samples[0]
        if isinstance(first, dict):
            return {
                key: torch.cat(
                    [self._ensure_batch_dim(sample[key]) for sample in samples],
                    dim=0,
                )
                for key in first
            }
        if isinstance(first, torch.Tensor):
            return torch.cat(
                [self._ensure_batch_dim(sample) for sample in samples],
                dim=0,
            )
        raise TypeError(f"Expected torch.Tensor or dict, got {type(first)}")

    @staticmethod
    def _shuffle_samples(samples):
        if isinstance(samples, dict):
            batch_size = next(iter(samples.values())).size(0)
            indices = torch.randperm(batch_size)
            return {key: value[indices] for key, value in samples.items()}

        indices = torch.randperm(samples.size(0))
        return samples[indices]

    def _split_samples(self, samples):
        if isinstance(samples, dict):
            first = next(iter(samples.values()))
            return [
                {key: value[start : start + self.batch_size] for key, value in samples.items()}
                for start in range(0, first.size(0), self.batch_size)
            ]
        return list(torch.split(samples, self.batch_size, dim=0))

    def close(self):
        for iterator in getattr(self, "iterators", []) or []:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        self.iterators = None
        for loader in self.loaders:
            close = getattr(loader, "close", None)
            if close is not None:
                close()
