import gc
import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run a tiny perplexity smoke through the real eval_ppl evaluator."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sshleifer/tiny-gpt2",
        help="Hugging Face causal LM model ID or local Mohawk checkpoint directory.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "hf", "mohawk"],
        default="auto",
        help="Model backend. auto uses Mohawk for local checkpoints with ComponentsConfig, otherwise Hugging Face.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="hello world from mohawk",
        help="Text used to build a temporary JSON dataset.",
    )
    parser.add_argument("--n_batches", type=int, default=1, help="Number of batches to evaluate.")
    parser.add_argument("--max_seq_len", type=int, default=16, help="Padded/truncated sequence length.")
    parser.add_argument("--batch_size", type=int, default=1, help="TorchDataLoader batch size.")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only use locally cached Hugging Face model/tokenizer files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--model-registration-module",
        action="append",
        default=[],
        metavar="MODULE",
        help="Import MODULE before model loading so it can register a custom Transformers model type. Repeat as needed.",
    )
    return parser


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    build_arg_parser().print_help()
    raise SystemExit(0)


import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataloaders import setup_dataloader
from evals.eval_api import EvalAPI
from utils.distributed import barrier, get_device, world_size
from utils.logging import logger
from utils.config import Config
from utils.build_model import import_model_registration_modules

class evaluator(EvalAPI):
    def __init__(self, cfg, DataConfig, n_batches, eval_type = "all_tokens", *args, **kwargs):
        super().__init__(*args, **kwargs, name="Perplexity")

        self.cfg = cfg
        self.n_batches = n_batches
        self.eval_type = eval_type
        self.dataloader = setup_dataloader(data_cfg=DataConfig, split="val")
        self.eos_token_id = self.dataloader.tokenizer.eos_token_id
        self.pad_token_id = self.dataloader.tokenizer.pad_token_id

    def is_better(self, current_best, new):
        return new["eval_score"] < current_best["eval_score"]

    @torch.no_grad()
    def __call__(self, wrapper_model, *args, **kwargs):
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        # Prepare model
        train_state = wrapper_model.model.training
        wrapper_model.model.eval()

        # We evaluate LLM using generated data and ground truth data
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="none", ignore_index=self.pad_token_id
        )
        total_correct = 0
        total_loss = 0
        num_tokens_seen = 0
        device = get_device()

        logger.info(f"[EVAL] Evaluating perplexity for {self.n_batches} batches")
        logger.info(f"[EVAL] Using:\n {self.dataloader}")
        for idx, batch in enumerate(self.dataloader):
            if idx >= self.n_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            input_ids = batch["input_ids"]
            position_ids = batch["position_ids"] if "position_ids" in batch else None

            # Forward pass with model
            model_outputs = wrapper_model(input_ids=input_ids, position_ids=position_ids)

            # Get logits
            if self.eval_type == "all_tokens":
                vocab_size = model_outputs.logits.shape[-1]
                logits = model_outputs.logits[:, :-1, :].contiguous().view(-1, vocab_size) # chop off the last token because we don't have a label for it
                labels = input_ids[..., 1:].contiguous().view(-1)
            elif self.eval_type == "last_token":
                last_token_idx = (input_ids[...,1:] == self.eos_token_id).to(torch.int).argmax(dim=1) - 1
                logits = model_outputs.logits[:, :-1, :].gather(
                    1, last_token_idx.view(-1,1,1).expand(-1,1,model_outputs.logits.size(-1))
                ).squeeze(1)
                labels = input_ids[...,1:].gather(1, last_token_idx.unsqueeze(1)).squeeze(1)
            else:
                raise ValueError(f"Unknown eval_type: {self.eval_type}")
            

            model_outputs = None


            # Setup outputs
            outputs = {}
            non_pad_mask = labels != self.pad_token_id
            actual_num_tokens = non_pad_mask.sum()

            # Perplexity
            loss = loss_fn(logits, labels).sum()
            outputs["ppl"] = torch.exp(loss / actual_num_tokens).item()
            outputs["loss"] = loss

            # Accuracy
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == labels)[non_pad_mask].sum()
            outputs["accuracy"] = (correct / actual_num_tokens).item()

            # accumulate
            total_loss += outputs["loss"].item()
            total_correct += correct.item()
            num_tokens_seen += actual_num_tokens.item()

            # Release memory
            logits = labels = None

        barrier()
        # Convert to tensors
        total_loss = torch.tensor(total_loss).to(device)
        total_correct = torch.tensor(total_correct).to(device)
        num_tokens_seen = torch.tensor(num_tokens_seen).to(device)

        # Reduce
        if world_size > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_tokens_seen, op=dist.ReduceOp.SUM)

        # Average
        total_loss = total_loss / num_tokens_seen
        total_correct = total_correct / num_tokens_seen
        avg_accuracy = total_correct.item()
        avg_perplexity = torch.exp(total_loss).item()

        # Back to Normal
        wrapper_model.model.train(train_state)

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        # Return
        return {
            "eval_score": avg_perplexity,
            "perplexity": avg_perplexity,
            "accuracy": avg_accuracy,
        }


class HfEvalWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, input_ids, position_ids=None, **kwargs):
        return self.model(input_ids=input_ids, position_ids=position_ids)


def _first_parameter_dtype(model):
    try:
        return str(next(model.parameters()).dtype)
    except StopIteration:
        return "unknown"


def _build_json_text_data_config(data_dir, tokenizer, max_seq_len, batch_size, local_files_only):
    return Config.from_dict(
        {
            "loaders": ["JSONIterableDataset", "Tokenize", "PaddingDataLoader", "TorchDataLoader"],
            "JSONIterableDataset": {"data_dir": data_dir},
            "Tokenize": {
                "tokenizer": tokenizer,
                "collate_type": "text",
                "local_files_only": local_files_only,
            },
            "PaddingDataLoader": {"max_seq_len": max_seq_len},
            "TorchDataLoader": {"batch_size": batch_size, "num_workers": 0},
        }
    )


def _is_mohawk_checkpoint(model_dir):
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return False
    try:
        cfg = Config.from_json(str(config_path))
    except Exception:
        return False
    return hasattr(cfg, "ComponentsConfig")


def _select_backend(args):
    if args.backend == "mohawk":
        return "mohawk"
    if args.backend == "hf":
        return "hf"
    return "mohawk" if _is_mohawk_checkpoint(args.model) else "hf"


def _get_nested_value(obj, *keys):
    for key in keys:
        if not hasattr(obj, key):
            return None
        obj = getattr(obj, key)
    return obj


def _tokenizer_source_from_mohawk_config(cfg):
    for keys in [
        ("TrainConfig", "tokenizer"),
        ("TrainDataConfig", "Tokenize", "tokenizer"),
        ("TeacherConfig", "tokenizer"),
    ]:
        tokenizer_source = _get_nested_value(cfg, *keys)
        if tokenizer_source:
            return tokenizer_source
    raise ValueError("Mohawk checkpoint config does not define a tokenizer.")


def run_hf_ppl_smoke(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        torch_dtype=torch.float32,
    ).to(device)
    wrapper = HfEvalWrapper(model)

    with tempfile.TemporaryDirectory() as tmp:
        data_path = f"{tmp}/data.json"
        with open(data_path, "w", encoding="utf-8") as handle:
            json.dump([{"text": args.text}], handle)
        data_cfg = _build_json_text_data_config(
            data_dir=tmp,
            tokenizer=args.model,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            local_files_only=args.local_files_only,
        )
        ppl_evaluator = evaluator(
            cfg=SimpleNamespace(),
            DataConfig=data_cfg,
            n_batches=args.n_batches,
        )
        result = ppl_evaluator(wrapper)

    result["model_dtype"] = _first_parameter_dtype(model)
    print(json.dumps(result, sort_keys=True))
    return result


def run_mohawk_ppl_smoke(args):
    from utils.init_model import lazy_init

    checkpoint_dir = Path(args.model)
    cfg = Config.from_json(str(checkpoint_dir / "config.json"))
    tokenizer_source = _tokenizer_source_from_mohawk_config(cfg)
    local_files_only = args.local_files_only or bool(
        _get_nested_value(cfg, "TrainDataConfig", "Tokenize", "local_files_only")
    )

    wrapper = lazy_init(
        cfg=cfg,
        details_cfg=cfg.TrainConfig,
        load_cfg=Config.from_dict({"model": [{"path": str(checkpoint_dir)}]}),
        mode="inference",
        components_cfg=cfg.ComponentsConfig,
    )

    with tempfile.TemporaryDirectory() as tmp:
        data_path = f"{tmp}/data.json"
        with open(data_path, "w", encoding="utf-8") as handle:
            json.dump([{"text": args.text}], handle)
        data_cfg = _build_json_text_data_config(
            data_dir=tmp,
            tokenizer=tokenizer_source,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            local_files_only=local_files_only,
        )
        ppl_evaluator = evaluator(
            cfg=cfg,
            DataConfig=data_cfg,
            n_batches=args.n_batches,
        )
        result = ppl_evaluator(wrapper)

    result["model_dtype"] = _first_parameter_dtype(wrapper.model)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    import_model_registration_modules(args.model_registration_module)
    if _select_backend(args) == "mohawk":
        run_mohawk_ppl_smoke(args)
    else:
        run_hf_ppl_smoke(args)
