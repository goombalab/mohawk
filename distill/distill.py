# Keep the above imports in order to run the code below
import os, argparse

from distill.main import distill as distillation_code
from utils.config import load_config
from utils.distributed import use_distributed, barrier, is_master
from utils.logging import logger, init_logging

job_id = os.environ.get("SLURM_JOBID", "0")

def set_config(cfg, save=True):
    # yaml -> dict
    configs_paths = cfg.split(",")
    configs_paths = [config_path.strip() for config_path in configs_paths]

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
        env_vars = os.environ
        for k, v in config.ManagementConfig.env_vars.items():
            os.environ[k] = str(v)

        with init_logging(config, wandb_id=wandb_id):
            config.ManagementConfig.wandb["id"] = wandb_id
            distillation_code(config)

        # Reset environment variables
        os.environ = env_vars
    
    barrier()


if __name__ == "__main__":
    distill()
