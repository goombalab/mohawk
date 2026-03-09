from dataloaders.BaseDataGenerator import BaseDataGenerator
from dataloaders.DataWrappers.TokenizedDataLoader import TokenizedDataLoader

class RoundRobinLoader(BaseDataGenerator, TokenizedDataLoader):
    """
    A data generator that iterates over multiple dataloaders in a round-robin fashion.
    That is, each time the generator is called, it will return the next batch from the next dataloader.
    """
    def __init__(self, train_data_configs, ratio=None, skip_samples=0, **kwargs):
        from dataloaders import setup_dataloader
        self.iterators = None
        self._idx = -1
        kwargs.pop("iterable", None)

        # check if ratio is provided
        self.ratio = [1] * len(train_data_configs) if ratio is None else ratio
        assert len(self.ratio) == len(train_data_configs), f"Number of ratios must match number of dataloaders: {len(train_data_configs)} != {len(self.ratio)}"

        # create dataloaders
        self.loaders = [setup_dataloader(config.TrainDataConfig) for config in train_data_configs]
        # self.loaders = []
        # for i, config in enumerate(train_data_configs):
        #     _skip_samples = int(skip_samples * (self.ratio[i] / sum(self.ratio)))
        #     config.TrainDataConfig.HFDataset.skip_samples = _skip_samples
        #     self.loaders.append(setup_dataloader(config.TrainDataConfig))

        # assert all dataloaders have the same tokenizer
        self._tokenizer = self.loaders[0].tokenizer
        assert all([self._tokenizer.name_or_path == loader.tokenizer.name_or_path for loader in self.loaders])

    def __iter__(self):
        # self.iterators = [iter(loader) for loader in self.loaders]
        # make em' reference the same iterator
        self.iterators = [[iter(loader)] * ratio for loader, ratio in zip(self.loaders, self.ratio)]
        self.iterators = sum(self.iterators, []) # flatten list
        return self
    
    def __next__(self):
        self._idx = (self._idx + 1) % len(self.iterators)
        return next(self.iterators[self._idx])

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif any([hasattr(loader, attr) for loader in self.loaders]):
            assert all([getattr(loader, attr) == getattr(self.loaders[0], attr) 
                        for loader in self.loaders if hasattr(loader, attr)]), f"Attribute '{attr}' has different values in round-robin loaders."
            return getattr(self.loaders[0], attr)
        raise AttributeError(f"No attribute '{attr}' found in round-robin loaders.")

    def __str__(self, tabs=0):
        return "---"*tabs + f"{self.__class__.__name__}\n" + "\n".join([loader.__str__(tabs=tabs+1) for loader in self.loaders])

    def state_dict(self):
        state = {
            "_idx": self._idx,
            "ratio": self.ratio
        }
        for i, loader in enumerate(self.loaders):
            state[f"loader_{i}"] = loader.state_dict()
        return state

    def load_state_dict(self, state_dict):
        for i, loader in enumerate(self.loaders):
            loader.load_state_dict(state_dict[f"loader_{i}"])
        self._idx = state_dict["_idx"]
        self.ratio = state_dict["ratio"]
