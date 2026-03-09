from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from .collate_fns import setup_collate_fn
from dataloaders.BaseDataWrapper import BaseDataWrapper
import os


class TokenizedDataLoader(DataLoader, BaseDataWrapper):
    """
    A data loader that tokenizes the input data using a specified tokenizer.
    """
    def __init__(
        self,
        iterable,
        tokenizer,
        collate_type,
        chat_template=None,
        add_generation_prompt=False,
        return_dict=False,
        **kwargs
    ):
        # setup tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        # setup special tokens
        if "pad_token" not in self._tokenizer.special_tokens_map:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # setup chat template
        if chat_template is not None:
            assert os.path.exists(chat_template), f"Chat template {chat_template} not found."
            chat_template = open(chat_template).read()
            chat_template = chat_template.replace('    ', '').replace('\n', '')
            # self._tokenizer.chat_template = chat_template
        else:
            chat_template = None

        # setup collate function
        self._collate_fn = setup_collate_fn(
            collate_type=collate_type, 
            tokenizer=self._tokenizer, 
            chat_template=chat_template, 
            add_generation_prompt=add_generation_prompt,
            return_dict=return_dict,
        )

        # call super class
        DataLoader.__init__(self, iterable, collate_fn=self._collate_fn, **kwargs)
        BaseDataWrapper.__init__(self, iterable, **kwargs)

    @property
    def tokenizer(self):
        return self._tokenizer
