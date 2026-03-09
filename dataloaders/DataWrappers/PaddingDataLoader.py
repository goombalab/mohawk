import torch
from typing import Iterable
from dataloaders.BaseDataWrapper import BaseDataWrapper
from utils.distributed import world_size

class PaddingDataLoader(BaseDataWrapper):
    """
    Packing sequence of samples into a single batch to exhaut max_seq_len.
    """

    def __init__(
        self, 
        iterable: Iterable,
        max_seq_len: int = 2048,
        **kwargs
    ):
        super(PaddingDataLoader, self).__init__(iterable, **kwargs)

        assert hasattr(iterable, "tokenizer"), "tokenizer not found in iterable"
        assert hasattr(iterable.tokenizer, "pad_token"), "pad_token not found in tokenizer.special_tokens_map"

        self.iterable = iterable
        self.pad_token = self.iterable.tokenizer.pad_token
        self.max_seq_len = max_seq_len
        self.pad_token_id = self.iterable.tokenizer.pad_token_id

    def pad_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Pad sequence to max_seq_len.
        """
        seq = seq.squeeze()
        pad_len = self.max_seq_len - seq.size(0)
        assert seq.dim() == 1, f"Expected 1D tensor, got {seq.dim()}D tensor"

        if pad_len <= 0: # Truncate sequence (no padding required)
            return seq[:self.max_seq_len]
        else: # Pad sequence
            return torch.cat([seq, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            

    def __iter__(self):
        """
        Iterate over this DataLoader, yielding batches.
        """
        for seq in self.iterable:
            if isinstance(seq, torch.Tensor):
                yield self.pad_sequence(seq)
            elif isinstance(seq, dict):
                yield {key: self.pad_sequence(seq[key]) for key in seq}
            else:
                raise TypeError(f"Expected torch.Tensor or dict, got {type(seq)}")
        
