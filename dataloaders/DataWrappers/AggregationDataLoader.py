from torch.utils.data import IterableDataset
from dataloaders.BaseDataWrapper import BaseDataWrapper
from utils.logging import logger

class AggregationDataLoader(BaseDataWrapper):
    def __init__(
            self, 
            iterable: IterableDataset, 
            aggregation_size: int = 10_000,
            **kwargs,
            ):
        super(AggregationDataLoader, self).__init__(iterable, **kwargs)
        self._dataloader = iterable
        self.aggregation_size = aggregation_size

    def __iter__(self):
        self.dataloader = iter(self._dataloader)
        while True:
            buffer = []
            try:
                logger.debug(f"Aggregating {self.aggregation_size} samples")
                for _ in range(self.aggregation_size):
                    buffer.append(next(self.dataloader))
                logger.debug(f"Aggregated {self.aggregation_size} samples")
            except StopIteration:
                if buffer:
                    yield from buffer
                break

            yield from buffer
