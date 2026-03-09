from streaming.base.util import clean_stale_shared_memory
from streaming.text.c4 import StreamingC4 as StreamingC4

from dataloaders.BaseDataGenerator import BaseDataGenerator

def C4DataLoader(data_cfg, split="train", **kwargs):

    dataset = StreamingC4(
        epoch_size=None,  # unbouded number of samples
        local=data_cfg.data_dir + f"/{split}",
        tokenizer_name=data_cfg.tokenizer,
        group_method="truncate",
        max_seq_len=data_cfg.get("max_seq_len", 256),
        batch_size=data_cfg.get("batch_size", 16),
        shuffle=data_cfg.get("shuffle", True),
        shuffle_seed=data_cfg.get("seed", 42),
        **kwargs,
    )


    clean_stale_shared_memory()
    return dataset
