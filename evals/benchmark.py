import warnings
warnings.simplefilter("ignore", FutureWarning)

import argparse
from contextlib import contextmanager
import os
import sys
from pathlib import Path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run lm-eval-harness benchmarks for a saved Mohawk or Hugging Face causal LM."
    )
    parser.add_argument("--dir", type=str, required=True, help="Directory of the saved model or Hugging Face model ID")
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated lm-eval task list")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_fewshot", type=int, default=None, help="Number of few-shot examples")
    parser.add_argument("--limit", type=float, default=None, help="Optional lm-eval sample limit for smoke runs")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only use locally cached Hugging Face model/tokenizer files and request offline lm-eval task data",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "hf", "mohawk"],
        default="auto",
        help="Benchmark backend. auto uses the Hugging Face backend when --dir is a loadable HF model.",
    )
    parser.add_argument("--device", type=str, default=None, help="Evaluation device. Defaults to cuda when available, else cpu.")
    parser.add_argument(
        "--model-registration-module",
        action="append",
        default=[],
        metavar="MODULE",
        help="Import MODULE before model loading so it can register a custom Transformers model type. Repeat as needed.",
    )
    return parser


if __name__ == "__main__":

    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        build_arg_parser().print_help()
        raise SystemExit(0)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# distributed
import torch.distributed as torch_dist
import utils.distributed as distributed_utils
from utils.distributed import local_rank, global_rank, world_size
from utils.distributed import use_distributed


# Hugging Face
from evals.eval_api import EvalAPI

# lm_eval
from lm_eval import simple_evaluate

# Other
from utils.logging import init_logging, logger, wandb
from utils.config import Config
from utils.build_model import get_tokenizer, import_model_registration_modules
from utils.init_model import lazy_init, eager_init
from utils.utils import catch_huggingface_http_errors


def aggregate_results(evaluation: dict, tasks: list):
    """
    Aggregates results from multiple runs.

    We follow:
    - https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about#task-evaluations-and-parameters
    """
    if "results" not in evaluation:
        return {}
    agg_results = {}
    priority = ["acc_norm,none", "acc,none", "contains,none", "exact_match,flexible-extract", 
                "exact_match,strict-match", "inst_level_strict_acc,none", "word_perplexity,none"]
    for task, results in evaluation["results"].items():
        if task not in tasks:
            continue # we take groups of tasks and not individual tasks
        elif task =='arc_easy':
            agg_results[task] = results["acc,none"]
            continue


        for name in priority:
            if name in results:
                agg_results[task] = results[name]
                break
        
        if task not in agg_results:
            agg_results.update({f"{task} ({name})": value for name, value in results.items() if name != 'alias'})
        
    results = pd.DataFrame([agg_results], index=["Result"]).transpose()
    results = results.sort_index()
    results.loc['AVG'] = results.mean(numeric_only=True)

    return results

def samples_to_file(evaluation: dict):
    if "samples" not in evaluation:
        return
    
    for task, samples in evaluation["samples"].items():
        with open(f"{wandb.dir}/{task}.txt", "w") as f:
            for sample in samples:
                f.write(f"{sample}\n\n")
    
    logger.info(f"Samples saved to {wandb.dir}")


def _tasks_from_arg(tasks: str) -> list:
    parsed_tasks = [task.strip() for task in tasks.split(",") if task.strip()]
    if not parsed_tasks:
        raise ValueError("At least one lm-eval task must be provided.")
    return parsed_tasks


class LocalHfCacheError(RuntimeError):
    pass


@contextmanager
def _local_hf_cache_mode(enabled: bool):
    if not enabled:
        yield
        return

    missing = object()
    env_names = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]
    previous_env = {name: os.environ.get(name) for name in env_names}
    datasets_module = None
    previous_datasets_offline = missing
    previous_datasets_hub_offline = missing
    previous_datasets_cache = missing
    hub_constants = None
    previous_hub_offline = None
    hub_http = None
    transformers_hub = None
    previous_transformers_offline = None

    try:
        for name in env_names:
            os.environ[name] = "1"
        try:
            import datasets as datasets_module

            previous_datasets_offline = getattr(datasets_module.config, "HF_DATASETS_OFFLINE", missing)
            if previous_datasets_offline is not missing:
                datasets_module.config.HF_DATASETS_OFFLINE = True
            previous_datasets_cache = getattr(datasets_module.config, "HF_DATASETS_CACHE", missing)
            if previous_datasets_cache is not missing and os.environ.get("HF_DATASETS_CACHE"):
                datasets_module.config.HF_DATASETS_CACHE = os.environ["HF_DATASETS_CACHE"]
            previous_datasets_hub_offline = getattr(datasets_module.config, "HF_HUB_OFFLINE", missing)
            if previous_datasets_hub_offline is not missing:
                datasets_module.config.HF_HUB_OFFLINE = True
        except Exception:
            datasets_module = None
        try:
            import huggingface_hub.constants as hub_constants
            import huggingface_hub.utils._http as hub_http

            previous_hub_offline = hub_constants.HF_HUB_OFFLINE
            hub_constants.HF_HUB_OFFLINE = True
            hub_http.reset_sessions()
        except Exception:
            hub_constants = None
            hub_http = None
        try:
            import transformers.utils.hub as transformers_hub

            previous_transformers_offline = transformers_hub._is_offline_mode
            transformers_hub._is_offline_mode = True
        except Exception:
            transformers_hub = None
        yield
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if datasets_module is not None and previous_datasets_offline is not missing:
            datasets_module.config.HF_DATASETS_OFFLINE = previous_datasets_offline
        if datasets_module is not None and previous_datasets_cache is not missing:
            datasets_module.config.HF_DATASETS_CACHE = previous_datasets_cache
        if datasets_module is not None and previous_datasets_hub_offline is not missing:
            datasets_module.config.HF_HUB_OFFLINE = previous_datasets_hub_offline
        if hub_constants is not None and previous_hub_offline is not None:
            hub_constants.HF_HUB_OFFLINE = previous_hub_offline
        if transformers_hub is not None and previous_transformers_offline is not None:
            transformers_hub._is_offline_mode = previous_transformers_offline
        if hub_http is not None:
            hub_http.reset_sessions()


def _resolve_hf_model_source(model_dir: str, local_files_only: bool = False) -> str:
    if not local_files_only or Path(model_dir).exists():
        return model_dir

    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_dir, local_files_only=True)
    except Exception as exc:
        raise LocalHfCacheError(
            f"Could not resolve {model_dir!r} from the local Hugging Face cache. "
            "Populate the cache first or rerun without --local_files_only."
        ) from exc


def _can_load_hf_config(model_dir: str, local_files_only: bool = False) -> bool:
    try:
        model_source = _resolve_hf_model_source(model_dir, local_files_only)
        AutoConfig.from_pretrained(model_source, local_files_only=local_files_only)
    except LocalHfCacheError:
        raise
    except Exception:
        return False
    return True


def _should_run_hf_backend(args) -> bool:
    if args.backend not in {"auto", "hf"}:
        return False
    return _can_load_hf_config(args.dir, args.local_files_only)


def _select_backend(args) -> str:
    if _should_run_hf_backend(args):
        return "hf"
    if args.backend == "hf":
        raise ValueError(f"Could not load {args.dir!r} as a Hugging Face model/config.")
    return "mohawk"


def _select_backend_for_cli(args) -> str:
    _tasks_from_arg(args.tasks)
    return _select_backend(args)


def _select_backend_or_exit(args) -> str:
    try:
        return _select_backend_for_cli(args)
    except (LocalHfCacheError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def _get_nested_value(obj, *keys):
    for key in keys:
        if not hasattr(obj, key):
            return None
        obj = getattr(obj, key)
    return obj


def _checkpoint_train_model_dtype(model_dir: str) -> str:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return "float32"
    try:
        checkpoint_cfg = Config.from_json(str(config_path))
    except Exception:
        return "float32"
    return _get_nested_value(checkpoint_cfg, "TrainConfig", "model_dtype") or "float32"


def _first_parameter_dtype(model):
    if model is None or not hasattr(model, "parameters"):
        return "unknown"
    try:
        return str(next(model.parameters()).dtype)
    except (StopIteration, TypeError, AttributeError):
        return "unknown"


def run_hf_benchmark(args):
    """
    Run lm-eval's built-in Hugging Face adapter on an already-initialized model.

    Initializing the model here avoids a compatibility issue between the
    installed lm_eval 0.4.11 adapter and this Transformers build, where lm_eval
    passes ``dtype=...`` into ``from_pretrained`` but GPT-2 expects
    ``torch_dtype=...``.
    """
    from lm_eval.models.huggingface import HFLM as LMEvalHFLM

    tasks = _tasks_from_arg(args.tasks)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_source = _resolve_hf_model_source(args.dir, args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        local_files_only=args.local_files_only,
        torch_dtype=torch.float32,
    ).eval()
    if device != "cpu":
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    wrapped_model = LMEvalHFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=args.batch_size,
        device=device,
        dtype=None,
    )
    with _local_hf_cache_mode(args.local_files_only):
        evaluation = simple_evaluate(
            model=wrapped_model,
            model_args=None,
            tasks=tasks,
            batch_size=args.batch_size,
            log_samples=False,
            device=device,
            limit=args.limit,
            gen_kwargs="",
            verbosity="ERROR",
            num_fewshot=args.num_fewshot,
            cache_requests=False,
            delete_requests_cache=False,
        )
    results = aggregate_results(evaluation, tasks)
    print(results)
    return results


class evaluator(EvalAPI):
    def __init__(
        self,
        tasks : list,
        cfg : Config = None,
        tokenizer : object = None,
        enable_tqdm : bool = False,
        distributed : bool = True,
        batch_size : int = 64,
        max_length : int = 2048,
        num_fewshot : int = None,
        log_samples : bool = False,
        limit : int = None,
        device : str = None,
        local_files_only : bool = False,
        *args : list,
        **kwargs : dict,
    ):
        super().__init__(*args, **kwargs, name="eval_lm")
        self.cfg = cfg
        self.tasks = tasks
        self.enable_tqdm = enable_tqdm
        self.distributed = distributed
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_fewshot = num_fewshot
        self.log_samples = log_samples
        self.limit = limit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.local_files_only = local_files_only
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
                raise Exception("Tokenizer not found")
        
    def is_better(self, current_best, new):
        """
        A higher lm_eval score is better.
        """
        return current_best["eval_score"] < new["eval_score"]

    def __call__(self, wrapper_model, *args, **kwargs):
        # import variables
        global global_rank, world_size

        # Prepare model
        train_state = wrapper_model.model.training
        wrapper_model.model.train(False)
        torch.cuda.empty_cache()

        # Wrap the model
        if not self.distributed:
            global_rank, world_size = 0, 1
            
        wrapper_model.device = torch.device(self.device)
        from evals.HFLM import HFLM

        wrapped_model = HFLM(
            pretrained=wrapper_model, 
            tokenizer=self.tokenizer, 
            backend="causal", 
            max_length=self.max_length,
            batch_size=self.batch_size,
            add_bos_token=True,
            )
        wrapped_model.accelerator = wrapper_model
        wrapped_model.accelerator.wait_for_everyone = (
            torch_dist.barrier if torch_dist.is_initialized() else (lambda: None)
        )
        wrapped_model._max_length = self.max_length
        wrapped_model._rank = global_rank
        wrapped_model._world_size = world_size

        logger.info(f"[BENCHMARK] Running benchmark:\n"
            f"- Tasks: {self.tasks}\n"
            f"- Batch size: {wrapped_model.batch_size}\n"
            f"- Tokenizer: {wrapped_model.tokenizer.name_or_path}\n"
            f"- max_length: {wrapped_model._max_length}\n"
            f"- num_fewshot: {self.num_fewshot}\n"
            f"- limit: {self.limit}\n"
            )

        with catch_huggingface_http_errors(retries=10, delay=20):
            # Run the benchmark
            results = _benchmark(
                    wrapped_model=wrapped_model, 
                    tasks=self.tasks,
                    limit=self.limit,
                    batch_size=self.batch_size, 
                    num_fewshot=self.num_fewshot,
                    log_samples=self.log_samples,
                    device=self.device,
                    local_files_only=self.local_files_only,
                    **kwargs
                )
            
        # Back to Normal
        torch.cuda.empty_cache()
        wrapper_model.model.train(train_state)

        return {
            "eval_score": results.loc['AVG'].mean(),
            "task_results": results
            }

# FSDP full-state checkpointing after wrapper-integrated eval needs normal
# tensors. `inference_mode` can leave FSDP internals as inference tensors.
@torch.no_grad()
def _benchmark(
    wrapped_model,
    tasks,
    log_samples,
    limit,
    num_fewshot=None,
    device=None,
    local_files_only=False,
    verbosity="ERROR",
    **kwargs,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with _local_hf_cache_mode(local_files_only):
        evaluation = simple_evaluate(
            model=wrapped_model,
            model_args=None,
            tasks=tasks,
            batch_size=wrapped_model.batch_size,
            log_samples=log_samples,
            device=device,
            limit=limit,
            gen_kwargs="",
            verbosity=verbosity,
            num_fewshot=num_fewshot,
            cache_requests=False,
            delete_requests_cache=False,
        )
    if world_size > 1:
        results = aggregate_results(evaluation, tasks) if global_rank == 0 else None
        tmp = [results if global_rank == 0 else None]
        torch_dist.broadcast_object_list(tmp, src=0)
        results = tmp[0]
    else:
        results = aggregate_results(evaluation, tasks)

    if global_rank == 0 and evaluation is not None:
        samples_to_file(evaluation)

    if world_size > 1:
        torch_dist.barrier()
    return results

def _run_mohawk_benchmark(args):
    parsed_tasks = _tasks_from_arg(args.tasks)
    local_files_only = getattr(args, "local_files_only", False)
    device = getattr(args, "device", None)

    logger.info(f"Running benchmark:\n"
                f"- World size: {world_size}\n"
                f"- Directory: {args.dir}\n"
                f"- Tasks: {args.tasks}\n"
                f"- Batch size: {args.batch_size}\n"
                f"- Num fewshot: {args.num_fewshot}\n"
                )

    # Initialize model
    wrapper = lazy_init(
    # wrapper = eager_init(
        details_cfg=cfg.TrainConfig,
        load_cfg=Config.from_dict({"model": [{"path": args.dir}]}),
        cfg=cfg,
        mode="inference",
    )
    print(f"Model dtype: {_first_parameter_dtype(getattr(wrapper, 'model', None))}")

    # Create evaluator object
    evaluator_obj = evaluator(
        tasks=parsed_tasks,
        cfg=None,
        batch_size=args.batch_size,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        tokenizer=get_tokenizer(args.dir, local_files_only=local_files_only),
        distributed=True if world_size > 1 else False,
        enable_tqdm=True,
        log_samples=False,
        device=device,
        local_files_only=local_files_only,
        )
    logger.info(evaluator_obj(wrapper_model=wrapper))


def main(args):
    distributed_backend = "nccl"
    if getattr(args, "device", None) == "cpu" and not torch.cuda.is_available():
        os.environ.setdefault("MOHAWK_ALLOW_CPU_TRAINING", "1")
    with distributed_utils.init_distributed(backend=distributed_backend):
        _run_mohawk_benchmark(args)
    

if __name__ == "__main__":
    """
    Example:
    python benchmark.py --dir /path/to/model/checkpoint
    OR
    PYTHONPATH=/path/to/mohawk python ./evals/benchmark.py --dir /path/to/model/checkpoint
    """
    # Parse arguments
    args = build_arg_parser().parse_args()
    import_model_registration_modules(args.model_registration_module)

    backend = _select_backend_or_exit(args)

    if backend == "hf":
        run_hf_benchmark(args)
        raise SystemExit(0)

    wandb_mode = os.environ.get("WANDB_MODE", "disabled")
    cfg = Config.from_dict({
        "ManagementConfig": {
            "wandb": {
                "key": os.environ.get("WANDB_API_KEY", "") if wandb_mode == "online" else "",
                "project": "distillation",
                "entity": os.environ.get("WANDB_ENTITY", "your-wandb-entity"),  # Use WANDB_ENTITY env var
                "mode": wandb_mode,
                "save_code": False,
            }
        },
        "DistillConfig": {
            "name": "benchmark",
        },
        "TrainConfig": {
            "dir": args.dir,
            "model_dtype": _checkpoint_train_model_dtype(args.dir),
            "activation_checkpointing": False,
            "mixed_precision": True,
            "compile_model": False,
        }
    })
    with init_logging(cfg):
        main(args)
