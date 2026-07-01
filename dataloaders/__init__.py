import inspect


def _missing_loader(name, exc):
    def raise_missing_dependency(*args, **kwargs):
        raise ImportError(
            f"{name} requires an optional dependency that is not installed: {exc}"
        ) from exc

    raise_missing_dependency.__name__ = name
    return raise_missing_dependency


# DATAGENERATORS
try:
    from dataloaders.DataGenerators.C4DataLoader import C4DataLoader
except ModuleNotFoundError as exc:
    C4DataLoader = _missing_loader("C4DataLoader", exc)

try:
    from dataloaders.DataGenerators.HFDataset import HFDataset
except ModuleNotFoundError as exc:
    HFDataset = _missing_loader("HFDataset", exc)

from dataloaders.DataGenerators.JSONIterableDataset import JSONIterableDataset

try:
    from dataloaders.DataGenerators.CopyingTaskDataset import CopyingTaskDataset
except ModuleNotFoundError as exc:
    CopyingTaskDataset = _missing_loader("CopyingTaskDataset", exc)

try:
    from dataloaders.DataGenerators.NeedleInHaystackDataset import NeedleInHaystackDataset
except ModuleNotFoundError as exc:
    NeedleInHaystackDataset = _missing_loader("NeedleInHaystackDataset", exc)

try:
    from dataloaders.DataGenerators.KVRetrieval import KVRetrieval
except ModuleNotFoundError as exc:
    KVRetrieval = _missing_loader("KVRetrieval", exc)

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
    "TokenizedDataLoader": TokenizedDataLoader,
    "AggregationDataLoader": AggregationDataLoader,
    "TorchDataLoader": TorchDataLoader,
    "PaddingDataLoader": PaddingDataLoader,
    "SequentialLoader": SequentialLoader,
    "CopyingTaskDataset": CopyingTaskDataset,
    "NeedleInHaystackDataset": NeedleInHaystackDataset,
    "KVRetrieval": KVRetrieval,
}

def filter_kwargs(cls, kwargs, include_var_keyword=False):
    """
    Filter the kwargs to only include the ones that are in the class __init__ method.
    """
    target = cls if inspect.isfunction(cls) else cls.__init__
    signature = inspect.signature(target)
    if include_var_keyword and any(
        param.kind == param.VAR_KEYWORD for param in signature.parameters.values()
    ):
        return dict(kwargs)
    accepted = {
        name
        for name, param in signature.parameters.items()
        if param.kind in {param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY}
        and name != "self"
    }
    return {k: v for k, v in kwargs.items() if k in accepted}

def setup_dataloader(data_cfg, **kwargs):

    final_loader = None
    for loader_name in data_cfg["loaders"]:
        loader_cls = DATASETS[loader_name]
        # Check if the loader is a composite loader
        _kwargs = filter_kwargs(loader_cls, kwargs)
        _kwargs.update(filter_kwargs(loader_cls, data_cfg))
        _kwargs.update(
            filter_kwargs(
                loader_cls,
                data_cfg[loader_name],
                include_var_keyword=True,
            )
        )
        # Instantiate the loader
        if inspect.isfunction(loader_cls) and "data_cfg" in inspect.signature(loader_cls).parameters:
            final_loader = loader_cls(data_cfg=data_cfg[loader_name], **_kwargs)
        else:
            final_loader = loader_cls(iterable=final_loader, **_kwargs)
    return final_loader
