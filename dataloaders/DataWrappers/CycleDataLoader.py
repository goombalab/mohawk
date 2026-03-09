from utils.logging import logger
from dataloaders.BaseDataWrapper import BaseDataWrapper
from torch.utils.data import IterableDataset

class CycleDataLoader(BaseDataWrapper):
    def __init__(self, iterable: IterableDataset, **kwargs):
        super(CycleDataLoader, self).__init__(iterable, **kwargs)

    def __iter__(self):
        while True:
            yield from self.iterable
            logger.warning("[CYCLE DATALOADER] Dataloader has been exhausted. Restarting...")
