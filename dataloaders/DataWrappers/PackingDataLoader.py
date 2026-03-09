import torch
from typing import Iterable
from dataloaders.BaseDataWrapper import BaseDataWrapper
from utils.logging import logger

class PackingDataLoader(BaseDataWrapper):
    """
    Packing sequence of samples into a single batch to exhaut max_seq_len.
    """

    def __init__(
        self, 
        iterable: Iterable,
        max_seq_len: int = 2048,
        **kwargs
    ):
        super(PackingDataLoader, self).__init__(iterable, **kwargs)
        self.iterable = iterable

        self.max_seq_len = max_seq_len
        self.bos_token = iterable.tokenizer.bos_token_id
        self.eos_token = iterable.tokenizer.eos_token_id
        self.pad_token = iterable.tokenizer.pad_token_id

        if self.bos_token is None:
            logger.warning("[PackingDataLoader] BOS token is not set. Using PAD token as BOS token.")
            self.bos_token = self.pad_token 

    def __iter__(self):
        """
        Iterate over this DataLoader, yielding batches.
        """
        sample = torch.tensor([], dtype=torch.long)
        position_ids = torch.tensor([], dtype=torch.long)
        for input_id in self.iterable:
            
            if hasattr(input_id, "keys"):
                input_id = input_id["input_ids"]
            if not isinstance(input_id, torch.Tensor):
                input_id = torch.tensor(input_id)
            if input_id.dim() > 1:
                input_id = input_id.squeeze(0)

            # Add bos token if not already present
            if input_id[0] != self.bos_token:
                input_id = torch.cat([torch.tensor([self.bos_token]), input_id], dim=0)

            # Remove pad tokens (from the end)
            input_id = input_id[input_id != self.pad_token]
            if self.pad_token == self.eos_token:
                input_id = torch.cat([input_id, torch.tensor([self.eos_token])], dim=0)

            # Note: NO NEED to add eos for the last sample as it is halved
            position_ids = torch.cat([position_ids, torch.arange(input_id.size(0), dtype=torch.long)], dim=0)[: self.max_seq_len]
            sample = torch.cat([sample, input_id], dim=0)[: self.max_seq_len]
            if len(sample) == self.max_seq_len:
                # yield sample, position_ids
                yield {"input_ids": sample, "position_ids": position_ids, "attention_mask": torch.ones_like(sample)}
                sample = torch.tensor([], dtype=torch.long)
                position_ids = torch.tensor([], dtype=torch.long)
