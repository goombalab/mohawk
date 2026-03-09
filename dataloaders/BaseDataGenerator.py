from torch.utils.data import IterableDataset

class BaseDataGenerator(IterableDataset):
    """
    Base class for all data generators.
    A Data Generator doesn't recieve an iterable dataset, but returns one.
    """
    def __init__(
        self,
        **kwargs
    ):
        super(BaseDataGenerator, self).__init__()

    @property
    def iterable(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError
    
    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError
    
    # get attribute from self or iterable
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif hasattr(self.iterable, attr):
            return getattr(self.iterable, attr)
        raise AttributeError(f"Neither TorchDataLoader nor iterable have the attribute '{attr}'")
    
    def __str__(self, tabs=0, **kwargs):
        return f"---"*tabs + f"{self.__class__.__name__}"
    
    def __repr__(self, *args, **kwargs):
        return self.__str__(*args, **kwargs)

