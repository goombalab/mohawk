# DATAGENERATORS
from dataloaders.DataGenerators.C4DataLoader import C4DataLoader
from dataloaders.DataGenerators.HFDataset import HFDataset
from dataloaders.DataGenerators.JSONIterableDataset import JSONIterableDataset
from dataloaders.DataGenerators.CopyingTaskDataset import CopyingTaskDataset
from dataloaders.DataGenerators.NeedleInHaystackDataset import NeedleInHaystackDataset
from dataloaders.DataGenerators.KVRetrieval import KVRetrieval

# DATAWRAPPERS
from dataloaders.DataWrappers.RoundRobinLoader import RoundRobinLoader
from dataloaders.DataWrappers.ShuffleLoader import ShuffleLoader
from dataloaders.DataWrappers.RandomDataLoader import RandomDataLoader
from dataloaders.DataWrappers.CycleDataLoader import CycleDataLoader
from dataloaders.DataWrappers.TorchDataLoader import TorchDataLoader
from dataloaders.DataWrappers.TokenizedDataLoader import TokenizedDataLoader
from dataloaders.DataWrappers.PaddingDataLoader import PaddingDataLoader
from dataloaders.DataWrappers.SequentialLoader import SequentialLoader
from dataloaders.DataWrappers.PackingDataLoader import PackingDataLoader
from dataloaders.DataWrappers.AggregationDataLoader import AggregationDataLoader


__all__ = ["setup_dataloader"]

DATASETS = {
    "C4DataLoader": C4DataLoader,
    "RandomDataLoader": RandomDataLoader,
    "JSONIterableDataset": JSONIterableDataset,
    "HFDataset": HFDataset,
    "CycleDataLoader": CycleDataLoader,
    "RoundRobinLoader": RoundRobinLoader,
    "ShuffleLoader": ShuffleLoader,
    "PackingDataLoader": PackingDataLoader,
    "Tokenize": TokenizedDataLoader,
    "AggregationDataLoader": AggregationDataLoader,
    "TorchDataLoader": TorchDataLoader,
    "PaddingDataLoader": PaddingDataLoader,
    "SequentialLoader": SequentialLoader,
    "CopyingTaskDataset": CopyingTaskDataset,
    "NeedleInHaystackDataset": NeedleInHaystackDataset,
    "KVRetrieval": KVRetrieval,
}

def filter_kwargs(cls, kwargs):
    """
    Filter the kwargs to only include the ones that are in the class __init__ method.
    """
    # Get the names of the arguments in the __init__ method
    init_args = cls.__init__.__code__.co_varnames
    # Filter the kwargs to only include the ones that are in the init_args
    return {k: v for k, v in kwargs.items() if k in init_args}

def setup_dataloader(data_cfg, **kwargs):

    final_loader = None
    for loader_name in data_cfg["loaders"]:
        loader_cls = DATASETS[loader_name]
        # Check if the loader is a composite loader
        _kwargs = filter_kwargs(loader_cls, kwargs)
        _kwargs.update(filter_kwargs(loader_cls, data_cfg))
        _kwargs.update(filter_kwargs(loader_cls, data_cfg[loader_name]))
        # Instantiate the loader
        final_loader = loader_cls(iterable=final_loader, **_kwargs)
    return final_loader
