import json
import os
from utils.distributed import world_size, global_rank
from dataloaders.BaseDataGenerator import BaseDataGenerator 


class JSONIterableDataset(BaseDataGenerator):
    def __init__(self, data_dir, **kwargs):
        self.json_dir = data_dir
        assert os.path.isdir(self.json_dir), f"Directory {self.json_dir} does not exist."
        self.json_files = [os.path.join(self.json_dir, f) for f in os.listdir(self.json_dir) if f.endswith('.json')]
        assert len(self.json_files) > 0, f"No JSON files found in {self.json_dir}."

        # Get the rank and world size in a distributed setup
        self.rank = global_rank
        self.world_size = world_size

        # Split the JSON files based on the rank 
        # (start from the rank-th file and take every world_size-th file)
        self.json_files = self.json_files[self.rank::self.world_size]

    def __iter__(self):
        for json_file in self.json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            for sample in data:
                yield sample
