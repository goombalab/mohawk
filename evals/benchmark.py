import warnings
warnings.simplefilter("ignore", FutureWarning)

import os
import pandas as pd
import torch

# distributed
import torch.distributed as torch_dist
from utils.distributed import local_rank, global_rank, world_size
from utils.distributed import use_distributed


# Hugging Face
from evals.eval_api import EvalAPI

# lm_eval
from lm_eval import simple_evaluate
from evals.HFLM import HFLM

# Other
from utils.logging import init_logging, logger, wandb
from utils.config import Config
from utils.build_model import get_tokenizer
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
            
        wrapper_model.device = f"cuda:{local_rank}"
        wrapped_model = HFLM(
            pretrained=wrapper_model, 
            tokenizer=self.tokenizer, 
            backend="causal", 
            max_length=self.max_length,
            batch_size=self.batch_size,
            add_bos_token=True,
            )
        wrapped_model.accelerator = wrapper_model
        wrapped_model.accelerator.wait_for_everyone = torch_dist.barrier
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
                    **kwargs
                )
            
        # Back to Normal
        torch.cuda.empty_cache()
        wrapper_model.model.train(train_state)

        return {
            "eval_score": results.loc['AVG'].mean(),
            "task_results": results
            }

@torch.inference_mode()
def _benchmark(
    wrapped_model,
    tasks,
    log_samples,
    limit,
    num_fewshot=None,
    verbosity="ERROR",
    **kwargs,
):

    evaluation = simple_evaluate(
        model=wrapped_model,
        model_args=(),
        tasks=tasks,
        batch_size=wrapped_model.batch_size,
        log_samples=log_samples,
        device=f"cuda:{local_rank}",
        limit=limit,
        gen_kwargs="",
        verbosity=verbosity,
        num_fewshot=num_fewshot,
        cache_requests=False,
        delete_requests_cache=True,
    )
    if world_size > 1:
        tmp = [evaluation]
        torch_dist.broadcast_object_list(tmp, src=0)
        evaluation = tmp[0]

    results = aggregate_results(evaluation, tasks)
    samples_to_file(evaluation)

    if world_size > 1:
        torch_dist.barrier()
    return results

@use_distributed(backend="nccl")
def main(args):

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
        load_cfgs=[Config.from_dict({"path": args.dir})],
        cfg=cfg,
        mode="inference",
    )

    # Create evaluator object
    evaluator_obj = evaluator(
        tasks=args.tasks.replace(" ", "").split(","),
        cfg=None,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        tokenizer=get_tokenizer(args.dir),
        distributed=True if world_size > 1 else False,
        enable_tqdm=True,
        log_samples=False,
        )
    logger.info(evaluator_obj(wrapper_model=wrapper))
    

if __name__ == "__main__":
    """
    Example:
    python benchmark.py --dir /path/to/model/checkpoint
    OR
    PYTHONPATH=/path/to/mohawk python ./evals/benchmark.py --dir /path/to/model/checkpoint
    """
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory of the saved model")
    parser.add_argument("--tasks", type=str, help="Tasks to evaluate on")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_fewshot", type=int, default=None, help="Number of fewshot examples")
    args = parser.parse_args()

    cfg = Config.from_dict({
        "ManagementConfig": {
            "wandb": {
                "key": os.environ.get("WANDB_API_KEY", ""),  # Use WANDB_API_KEY env var
                "project": "distillation",
                "entity": os.environ.get("WANDB_ENTITY", "your-wandb-entity"),  # Use WANDB_ENTITY env var
            }
        },
        "DistillConfig": {
            "name": "benchmark",
        },
        "TrainConfig": {
            "dir": args.dir,
            "model_dtype": "float32",
            "activation_checkpointing": False,
            "mixed_precision": True,
            "compile_model": False,
        }
    })
    with init_logging(cfg):
        main(args)
