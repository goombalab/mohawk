import os

# os.environ["OMP_NUM_THREADS"] = "8"
import torch
import torch.distributed as dist
from contextlib import contextmanager

REQUIRED_ENV_VARS = [
    "LOCAL_RANK",
    "WORLD_SIZE",
    "RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
]


os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
os.environ["RANK"] = os.environ.get("RANK", "0")
os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
world_size = int(os.environ["WORLD_SIZE"])
global_rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
is_master = global_rank == 0

def details():
    return {
        "world_size": int(os.environ["WORLD_SIZE"]),
        "global_rank": int(os.environ["RANK"]),
        "local_rank": int(os.environ["LOCAL_RANK"]),
        "is_master": int(os.environ["RANK"]) == 0,
    }

def use_distributed(*cm_args, **cm_kwargs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with init_distributed(*cm_args, **cm_kwargs):
                return func(*args, **kwargs)
        return wrapper
    return decorator

@contextmanager
def init_distributed(backend='nccl', *args, **kwargs):
    """
    Context manager for initializing and destroying the distributed process group.

    Args:
        backend (str): The backend to use ('nccl', 'gloo', etc.).
        init_method (str): URL specifying how to initialize the process group.
        **kwargs: Additional arguments for `init_process_group`.

    Usage:
        with init_distributed(backend='nccl'):
            # Your distributed code here
    """
    assert not torch.distributed.is_initialized(), "Distributed is already initialized."
    assert all([env_var in os.environ for env_var in REQUIRED_ENV_VARS]), "Environment variables not set. Have you run `torchrun`?"

    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disable tokenizers parallelism
    os.environ["OMP_NUM_THREADS"] = "8" # Set number of threads for OpenMP
    os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1") # Set world size
    os.environ["RANK"] = os.environ.get("RANK", "0") # Set rank
    os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0") # Set local rank
    os.environ['CURL_CA_BUNDLE'] = '' # Disable SSL verification for HTTP requests (for Hugging Face models)

    if local_rank == 0:
        print(f"[DISTRIBUTED] Information:"
              f"\n\t- WORLD_SIZE={world_size}"
              f"\n\t- MASTER_ADDR={os.environ['MASTER_ADDR']}"
              f"\n\t- MASTER_PORT={os.environ['MASTER_PORT']}")

    # Set device (MUST BE DONE BEFORE INITIALIZING DISTRIBUTED)
    _details = details()
    torch.cuda.set_device(_details["local_rank"] % torch.cuda.device_count())

    if world_size == 1:
        yield
        return

    if "timeout" not in kwargs:
        # Set timeout to 4 hours (watchdog will kill the job if it hangs) 
        from datetime import timedelta
        kwargs["timeout"] = timedelta(seconds=60*60*4)

    try:
        dist.init_process_group(backend=backend, *args, **kwargs)
        yield
    except Exception as e:
        print(30*"=")
        print(f"[DISTRIBUTED] Error at rank {global_rank}:\n") # {e}")
        import traceback
        print(traceback.format_exc())
        print(30*"=")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            dist.destroy_process_group()
            if is_master:
                print("[DISTRIBUTED] Process group destroyed.")

def barrier():
    """
    Barrier for all processes in the group.
    """
    if world_size > 1:
        torch.distributed.barrier()
