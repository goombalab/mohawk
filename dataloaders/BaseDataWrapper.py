from torch.utils.data import IterableDataset

class BaseDataWrapper(IterableDataset):
    """
    Base class for all data wrappers.
    A Data Weapper receives an iterable dataset, and returns another iterable dataset.
    This is useful for wrapping an iterable dataset with a data loader.
    The data loader can be used to add additional functionality, such as collate functions,
    batching, and shuffling.
    """
    def __init__(
        self,
        iterable: IterableDataset,
        **kwargs
    ):
        super(BaseDataWrapper, self).__init__()
        self.iterable = iterable

    # get attribute from self or iterable
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif hasattr(self.iterable, attr):
            return getattr(self.iterable, attr)
        raise AttributeError(f"Neither TorchDataLoader nor iterable have the attribute '{attr}'")

    def state_dict(self):
        """
        Data wrappers are stateless, so we just return the state dict of the iterable.
        """
        return self.iterable.state_dict()
        
    def __str__(self, tabs=0, **kwargs):
        return f"---"*tabs + f"{self.__class__.__name__}\n" + self.iterable.__str__(tabs=tabs+1)
    
    def __repr__(self, *args, **kwargs):
        return self.__str__(*args, **kwargs)
