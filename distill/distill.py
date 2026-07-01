# Keep the above imports in order to run the code below
import os, argparse

from distill.main import distill as distillation_code
from utils.config import load_config
from utils.distributed import use_distributed, barrier, is_master
from utils.logging import logger, init_logging

job_id = os.environ.get("SLURM_JOBID", "0")
_MISSING_ENV = object()


def _apply_env_overrides(env_vars):
    previous = {}
    for key, value in env_vars.items():
        previous[key] = os.environ.get(key, _MISSING_ENV)
        if value == "" and previous[key] not in (_MISSING_ENV, ""):
            continue
        os.environ[key] = str(value)
    return previous


def _restore_env_overrides(previous):
    for key, value in previous.items():
        if value is _MISSING_ENV:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

def set_config(cfg, save=True):
    # yaml -> dict
    configs_paths = cfg.split(",")
    configs_paths = [config_path.strip() for config_path in configs_paths if config_path.strip()]
    if not configs_paths:
        raise ValueError("At least one config path must be provided.")

    configs = []
    for config_path in configs_paths:

        # Load config
        config = load_config(
            config_path=config_path, 
            CONSTANTS={
                "slurm_job_id": job_id,
                # "wandb_id": wandb.run.id
            })

        # save config
        if is_master and save:
            os.makedirs(config.ManagementConfig.paths.save_dir, exist_ok=True)
            config.save(path=os.path.join(config.ManagementConfig.paths.save_dir, "config.yaml"))

        # save config
        configs.append(config)

    return configs

@use_distributed(backend="nccl")
def distill():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    configs = set_config(args.config)
    wandb_id = None

    for config in configs:

        barrier()   
        # Set environment variables
        previous_env = _apply_env_overrides(config.ManagementConfig.env_vars)

        try:
            with init_logging(config, wandb_id=wandb_id):
                config.ManagementConfig.wandb["id"] = wandb_id
                distillation_code(config)
        finally:
            # Restore only values this config overrode. Clearing the entire
            # process environment can interfere with native-library teardown.
            _restore_env_overrides(previous_env)
    
    barrier()


if __name__ == "__main__":
    distill()
