from types import SimpleNamespace
import json

import pytest

torch = pytest.importorskip("torch")


def test_torch_dataloader_wraps_tensor_batches_as_input_ids():
    from torch.utils.data import IterableDataset
    from dataloaders.DataWrappers.TorchDataLoader import TorchDataLoader

    class TinyIterable(IterableDataset):
        def __iter__(self):
            yield torch.tensor([1, 2, 3])

    loader = TorchDataLoader(TinyIterable(), batch_size=1, num_workers=0)
    batch = next(iter(loader))

    assert set(batch) == {"input_ids"}
    assert batch["input_ids"].shape == (1, 3)


def test_torch_dataloader_iterator_close_reaches_wrapped_iterable():
    from torch.utils.data import IterableDataset
    from dataloaders.DataWrappers.TorchDataLoader import TorchDataLoader

    class ClosingIterable(IterableDataset):
        def __init__(self):
            self.iterator_closed = False
            self.close_called = False

        def __iter__(self):
            try:
                yield torch.tensor([1, 2, 3])
                yield torch.tensor([4, 5, 6])
            finally:
                self.iterator_closed = True

        def close(self):
            self.close_called = True

    iterable = ClosingIterable()
    loader = TorchDataLoader(iterable, batch_size=1, num_workers=0)
    iterator = iter(loader)

    assert next(iterator)["input_ids"].shape == (1, 3)

    iterator.close()
    loader.close()

    assert iterable.iterator_closed
    assert iterable.close_called


def test_base_data_wrapper_close_tolerates_leaf_generator_without_iterable():
    from dataloaders.BaseDataGenerator import BaseDataGenerator
    from dataloaders.DataWrappers.CycleDataLoader import CycleDataLoader
    from dataloaders.DataWrappers.PaddingDataLoader import PaddingDataLoader

    class LeafGenerator(BaseDataGenerator):
        tokenizer = SimpleNamespace(pad_token="<pad>", pad_token_id=0)

        def __iter__(self):
            yield torch.tensor([1, 2, 3])

    loader = CycleDataLoader(PaddingDataLoader(LeafGenerator(), max_seq_len=3))

    assert torch.equal(next(iter(loader)), torch.tensor([1, 2, 3]))
    loader.close()


def test_leaf_generator_missing_attribute_raises_attribute_error():
    from dataloaders.BaseDataGenerator import BaseDataGenerator

    class LeafGenerator(BaseDataGenerator):
        pass

    with pytest.raises(AttributeError):
        getattr(LeafGenerator(), "missing_attribute")


def test_padding_dataloader_pads_and_truncates_tensor_sequences():
    from torch.utils.data import IterableDataset
    from dataloaders.DataWrappers.PaddingDataLoader import PaddingDataLoader

    class TinyIterable(IterableDataset):
        tokenizer = SimpleNamespace(pad_token="<pad>", pad_token_id=0)

        def __iter__(self):
            yield torch.tensor([1, 2])
            yield torch.tensor([1, 2, 3, 4])

    loader = PaddingDataLoader(TinyIterable(), max_seq_len=3)
    iterator = iter(loader)

    assert torch.equal(next(iterator), torch.tensor([1, 2, 0]))
    assert torch.equal(next(iterator), torch.tensor([1, 2, 3]))


def test_cycle_dataloader_restarts_exhausted_iterable():
    from torch.utils.data import IterableDataset
    from dataloaders.DataWrappers.CycleDataLoader import CycleDataLoader

    class TinyIterable(IterableDataset):
        def __iter__(self):
            yield {"input_ids": torch.tensor([1])}

    iterator = iter(CycleDataLoader(TinyIterable()))

    assert torch.equal(next(iterator)["input_ids"], torch.tensor([1]))
    assert torch.equal(next(iterator)["input_ids"], torch.tensor([1]))


def test_aggregation_dataloader_yields_buffered_items_in_order():
    from torch.utils.data import IterableDataset
    from dataloaders.DataWrappers.AggregationDataLoader import AggregationDataLoader

    class TinyIterable(IterableDataset):
        def __iter__(self):
            yield {"input_ids": torch.tensor([1])}
            yield {"input_ids": torch.tensor([2])}
            yield {"input_ids": torch.tensor([3])}

    loader = AggregationDataLoader(TinyIterable(), aggregation_size=2)
    values = [int(batch["input_ids"][0]) for batch in loader]

    assert values == [1, 2, 3]


def test_shuffle_loader_supports_train_data_config_dict_batches(tmp_path, monkeypatch):
    import dataloaders
    from dataloaders.DataWrappers.ShuffleLoader import ShuffleLoader
    from utils.config import Config

    source_a = tmp_path / "source_a.yaml"
    source_b = tmp_path / "source_b.yaml"
    source_a.write_text("TrainDataConfig:\n  name: source_a\n")
    source_b.write_text("TrainDataConfig:\n  name: source_b\n")

    class TinyLoader:
        tokenizer = SimpleNamespace(name_or_path="tiny-tokenizer")
        max_seq_len = 2

        def __init__(self, value):
            self.value = value

        def __iter__(self):
            yield {"input_ids": torch.tensor([[self.value, self.value]])}

    def fake_setup_dataloader(data_cfg):
        return TinyLoader(1 if data_cfg.name == "source_a" else 2)

    monkeypatch.setattr(dataloaders, "setup_dataloader", fake_setup_dataloader)
    cfg = Config.from_dict(
        {
            "loaders": [str(source_a), str(source_b)],
            "batch_size": 1,
            "max_seq_len": 2,
        }
    )

    loader = ShuffleLoader(data_cfg=cfg)
    batches = [next(iter(loader)), next(loader)]
    values = {tuple(batch["input_ids"].reshape(-1).tolist()) for batch in batches}

    assert values == {(1, 1), (2, 2)}
    assert all(tuple(batch["input_ids"].shape) == (1, 2) for batch in batches)
    assert loader.tokenizer.name_or_path == "tiny-tokenizer"
    assert loader.max_seq_len == 2


def test_packing_dataloader_returns_training_dict():
    from torch.utils.data import IterableDataset
    from dataloaders.DataWrappers.PackingDataLoader import PackingDataLoader

    class TinyIterable(IterableDataset):
        tokenizer = SimpleNamespace(bos_token_id=10, eos_token_id=11, pad_token_id=0)

        def __iter__(self):
            yield torch.tensor([1, 2])
            yield torch.tensor([3, 4])

    loader = PackingDataLoader(TinyIterable(), max_seq_len=4)
    batch = next(iter(loader))

    assert set(batch) == {"input_ids", "position_ids", "attention_mask"}
    assert batch["input_ids"].shape == (4,)


@pytest.mark.hf
@pytest.mark.slow
def test_setup_dataloader_constructs_local_json_tokenizer_pipeline(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    from dataloaders import setup_dataloader
    from utils.config import Config

    model_id = "sshleifer/tiny-gpt2"
    try:
        transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except OSError as exc:
        pytest.skip(f"{model_id} is not available in the local Hugging Face cache: {exc}")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    (tmp_path / "data.json").write_text(json.dumps([{"text": "hello world"}]))
    cfg = Config.from_dict(
        {
            "loaders": ["JSONIterableDataset", "Tokenize"],
            "JSONIterableDataset": {"data_dir": str(tmp_path)},
            "Tokenize": {
                "tokenizer": model_id,
                "collate_type": "text",
                "local_files_only": True,
            },
        }
    )

    loader = setup_dataloader(cfg)
    batch = next(iter(loader))

    assert tuple(batch.shape) == (1, 2)
    assert batch.dtype == torch.int64


def test_hf_dataset_state_tracks_consumed_samples_for_resume(monkeypatch, tmp_path):
    hfd_module = pytest.importorskip("dataloaders.DataGenerators.HFDataset")
    HFDataset = hfd_module.HFDataset
    cache_dir = tmp_path / "hf-datasets-cache"

    class FakeSplit:
        def __init__(self, values):
            self.values = list(values)
            self.shuffle_args = None

        def shuffle(self, seed, buffer_size):
            self.shuffle_args = (seed, buffer_size)
            return self

        def skip(self, n):
            skipped = FakeSplit(self.values[n:])
            skipped.shuffle_args = self.shuffle_args
            return skipped

        def __iter__(self):
            for value in self.values:
                yield {"text": value}

    def fake_load_dataset(path, name=None, streaming=False, cache_dir=None):
        assert path == "fake/public"
        assert name is None
        assert streaming is True
        assert cache_dir == str(cache_dir_path)
        return {"train": FakeSplit(["alpha", "beta", "gamma"])}

    cache_dir_path = cache_dir
    monkeypatch.setenv("HF_DATASETS_CACHE", str(cache_dir_path))
    monkeypatch.setattr(hfd_module.datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        hfd_module.datasets.distributed,
        "split_dataset_by_node",
        lambda dataset, rank, world_size: dataset,
    )

    dataset = HFDataset(path="fake/public", streaming=True)
    iterator = iter(dataset)

    assert dataset.dataset.shuffle_args == (42, 10000)
    assert next(iterator) == {"text": "alpha"}
    assert dataset.state_dict()["_index"] == 1
    assert next(iterator) == {"text": "beta"}
    assert dataset.state_dict()["_index"] == 2

    resumed = HFDataset(path="fake/public", streaming=True, shuffle_buffer_size=2)
    resumed.load_state_dict(dataset.state_dict())

    assert resumed.dataset.shuffle_args == (42, 10000)
    assert next(iter(resumed)) == {"text": "gamma"}
    assert resumed.state_dict()["_index"] == 3


def test_hf_dataset_allows_streaming_shuffle_buffer_override(monkeypatch, tmp_path):
    hfd_module = pytest.importorskip("dataloaders.DataGenerators.HFDataset")
    HFDataset = hfd_module.HFDataset
    cache_dir_path = tmp_path / "hf-datasets-cache"

    class FakeSplit:
        def __init__(self):
            self.shuffle_args = None

        def shuffle(self, seed, buffer_size):
            self.shuffle_args = (seed, buffer_size)
            return self

        def skip(self, n):
            return self

        def __iter__(self):
            yield {"text": "alpha"}

    fake_split = FakeSplit()

    def fake_load_dataset(path, name=None, streaming=False, cache_dir=None):
        assert cache_dir == str(cache_dir_path)
        return {"train": fake_split}

    monkeypatch.setenv("HF_DATASETS_CACHE", str(cache_dir_path))
    monkeypatch.setattr(
        hfd_module.datasets,
        "load_dataset",
        fake_load_dataset,
    )
    monkeypatch.setattr(
        hfd_module.datasets.distributed,
        "split_dataset_by_node",
        lambda dataset, rank, world_size: dataset,
    )

    dataset = HFDataset(path="fake/public", streaming=True, shuffle_buffer_size=2)

    assert dataset.dataset.shuffle_args == (42, 2)


def test_hf_dataset_forwards_and_restores_dataset_kwargs(monkeypatch, tmp_path):
    hfd_module = pytest.importorskip("dataloaders.DataGenerators.HFDataset")
    HFDataset = hfd_module.HFDataset
    from dataloaders import setup_dataloader
    from utils.config import Config
    calls = []

    class FakeSplit:
        def shuffle(self, seed, buffer_size):
            return self

        def __iter__(self):
            yield {"text": "alpha"}

    def fake_load_dataset(path, name=None, streaming=False, cache_dir=None, **kwargs):
        calls.append(
            {
                "path": path,
                "name": name,
                "streaming": streaming,
                "cache_dir": cache_dir,
                **kwargs,
            }
        )
        return {"train": FakeSplit()}

    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf-datasets-cache"))
    monkeypatch.setattr(hfd_module.datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        hfd_module.datasets.distributed,
        "split_dataset_by_node",
        lambda dataset, rank, world_size: dataset,
    )

    data_files = {
        "train": [
            "https://huggingface.co/datasets/example/data/resolve/main/train.parquet"
        ]
    }
    dataset = HFDataset(
        path="parquet",
        streaming=True,
        iterable=None,
        data_files=Config.from_dict(data_files),
        revision="main",
    )
    state = dataset.state_dict()

    assert calls[-1]["data_files"] == data_files
    assert calls[-1]["revision"] == "main"
    assert "iterable" not in calls[-1]
    assert state["dataset_kwargs"] == {
        "data_files": data_files,
        "revision": "main",
    }

    resumed = HFDataset(path="parquet", streaming=True)
    resumed.load_state_dict(state)

    assert calls[-1]["data_files"] == data_files
    assert calls[-1]["revision"] == "main"
    assert resumed.dataset_kwargs == state["dataset_kwargs"]

    configured = setup_dataloader(
        Config.from_dict(
            {
                "loaders": ["HFDataset"],
                "HFDataset": {
                    "path": "parquet",
                    "streaming": True,
                    "data_files": data_files,
                    "revision": "main",
                },
            }
        )
    )

    assert calls[-1]["data_files"] == data_files
    assert calls[-1]["revision"] == "main"
    assert configured.dataset_kwargs == state["dataset_kwargs"]


def test_hf_dataset_close_closes_active_streaming_iterator(monkeypatch, tmp_path):
    hfd_module = pytest.importorskip("dataloaders.DataGenerators.HFDataset")
    HFDataset = hfd_module.HFDataset
    cache_dir_path = tmp_path / "hf-datasets-cache"
    closed = []

    class FakeSplit:
        def shuffle(self, seed, buffer_size):
            return self

        def __iter__(self):
            try:
                yield {"text": "alpha"}
                yield {"text": "beta"}
            finally:
                closed.append(True)

    def fake_load_dataset(path, name=None, streaming=False, cache_dir=None):
        assert cache_dir == str(cache_dir_path)
        return {"train": FakeSplit()}

    monkeypatch.setenv("HF_DATASETS_CACHE", str(cache_dir_path))
    monkeypatch.setattr(hfd_module.datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        hfd_module.datasets.distributed,
        "split_dataset_by_node",
        lambda dataset, rank, world_size: dataset,
    )

    dataset = HFDataset(path="fake/public", streaming=True)
    iterator = iter(dataset)

    assert next(iterator) == {"text": "alpha"}

    dataset.close()

    assert closed == [True]
    assert dataset._iterator is None
