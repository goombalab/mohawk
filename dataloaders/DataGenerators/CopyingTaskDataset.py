import random
from utils.logging import logger
from dataloaders.BaseDataGenerator import BaseDataGenerator 
from utils.build_model import get_tokenizer
import torch

import random

class CopyingTaskDataset(BaseDataGenerator):
    def __init__(
        self,
        tokenizer_name: str,
        max_seq_len: int = 512,
        local_files_only: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.tokenizer = get_tokenizer(tokenizer_name, local_files_only=local_files_only)

        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        assert self.pad_token_id is not None, "Either pad_token_id or eos_token_id should be defined"

    @staticmethod
    def to_string(elements):
        elements_str = ','.join(map(str, elements))
        return (
            f"Instruction: Remember the following numbers.\n"
            f"Context: {elements_str}\n"
            f"Response: {elements_str}"
        )
    
    def to_tokens(self, elements):
        return self.tokenizer(self.to_string(elements), return_tensors="pt")["input_ids"].squeeze()
    
    def pad_tokens(self, tokens):
        return torch.cat([tokens, torch.tensor([self.pad_token_id] * (self.max_seq_len - tokens.size(0)))])

    def __iter__(self):
        while True:
            elements = []
            while len(self.to_tokens(elements)) < self.max_seq_len:
                elements.append(random.randint(0, 9))
            elements.pop()

            yield self.pad_tokens(self.to_tokens(elements))

class KVRetrival(BaseDataGenerator):
    def __init__(self, tokenizer_name: str, max_seq_len: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.tokenizer = get_tokenizer(tokenizer_name)

    @staticmethod
    def to_string(numbers):
        return (
            f"Instruction: Remember the following numbers.\n"
            f"Context: {numbers}\n"
            f"Response: {numbers}"
        )

    def __iter__(self):
        while True:
            elements = {}
            counter = 0
            while len(self.to_string(elements)) < self.max_seq_len:
                elements[counter] = random.randint(0, 9)
                counter += 1
            elements.pop()
            yield self.to_string(elements)
