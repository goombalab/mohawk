import torch
from utils.build_model import get_tokenizer


class RandomDataLoader:
    def __init__(self, data_cfg, **kwargs):
        self.tokenizer = get_tokenizer(
            data_cfg.tokenizer,
            local_files_only=data_cfg.get("local_files_only", False),
        )
        self.vocab_size = self.tokenizer.vocab_size
        special_tokens_map = self.tokenizer.special_tokens_map
        for key, value in special_tokens_map.items():
            setattr(self, key, self.tokenizer.encode(value)[-1])

        split = kwargs.pop("split", "train")

        self.max_seq_len = data_cfg.max_seq_len
        self.n_tokens = data_cfg[split]["n_tokens"]
        self.num_samples = int(self.n_tokens // self.max_seq_len)
        self.batch_size = data_cfg[split]["batch_size"]

    def __iter__(self):
        num_batches = int(self.num_samples // self.batch_size)
        for _ in range(num_batches):
            yield {
                "input_ids": torch.randint(
                    0, self.vocab_size, (self.batch_size, self.max_seq_len)
                )
            }

    def __len__(self):
        return self.num_samples
