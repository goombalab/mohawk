from torch.utils.data import IterableDataset
from typing import List


class FilterDataset(IterableDataset):
    def __init__(self, dataloader: IterableDataset, filters: List[str]):
        self._dataloader = dataloader
        # Precompile the condition strings into lambda functions
        self.filters = [eval(f'lambda item: {condition}') for condition in filters]

    def __iter__(self):
        for item in self._dataloader:
            # Check if all conditions are satisfied
            if all(condition(item) for condition in self.filters):
                yield item
        
