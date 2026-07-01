import os

import torch
from utils.logging import logger
from utils.wsd_scheduler import get_wsd_scheduler

try:
    from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
except ModuleNotFoundError:
    LinearWarmupCosineAnnealingLR = None


def setup_scheduler(optimizer, opt_cfg, total_grad_steps):
    scheduler_cfg = opt_cfg.scheduler

    # Total Grad Steps
    # total_grad_steps = int(cfg.TrainConfig["n_batches"] / cfg.TrainConfig["accumulation_steps"])

    # Set warmup_steps
    warmup_steps = scheduler_cfg.get("warmup_steps", 0)
    assert type(warmup_steps) in [int, float], "warmup_steps must be a float or int."
    if 0 < warmup_steps < 1:
        warmup_steps = int(warmup_steps * total_grad_steps)
    else:
        warmup_steps = int(warmup_steps)

    # Set decay_steps
    decay_steps = scheduler_cfg.get("decay_steps", 0)
    assert type(decay_steps) in [int, float], "decay_steps must be a float or int."
    if 0 < decay_steps < 1:
        decay_steps = int(decay_steps * total_grad_steps)
    else:
        decay_steps = int(decay_steps)

    assert (
        0 <= warmup_steps + decay_steps < total_grad_steps
    ), "Warmup + Decay steps must be between 0 and total_grad_steps."


    logger.info(
        f"[SCHEDULER] Information: \
        \n - Warmup Steps: {warmup_steps} ({(warmup_steps/total_grad_steps)*100:.2f}% of total steps) \
        \n - Decay Steps: {decay_steps} ({(decay_steps/total_grad_steps)*100:.2f}% of total steps) \
        \n - Total Grad Steps: {total_grad_steps}"
    )
    
    # Initialize scheduler
    if scheduler_cfg["name"] == "cosine":
        if LinearWarmupCosineAnnealingLR is None:
            raise ImportError(
                "scheduler.name='cosine' requires the optional 'lightning-bolts' "
                "package that provides pl_bolts. Install it or use scheduler.name='wsd'/'constant'."
            )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_steps,
            max_epochs=total_grad_steps,
            warmup_start_lr=scheduler_cfg["warmup_start_lr"],
            eta_min=scheduler_cfg["min_lr"],
        )
    elif scheduler_cfg["name"] == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.0)
    elif scheduler_cfg["name"] == "wsd":
        scheduler = get_wsd_scheduler(
            optimizer,
            lr=opt_cfg.lr,
            warmup_start_lr=scheduler_cfg["warmup_start_lr"],
            warmup_end=warmup_steps,
            min_lr=scheduler_cfg["min_lr"],
            standard_end=total_grad_steps - decay_steps,
            decay_end=total_grad_steps,
            phase="entire",
            decay_function="linear",
        )
    elif scheduler_cfg["name"] == "wsd_train":
        if warmup_steps == 0:
            logger.info(
                "[SCHEDULER] warmup_epochs is None. Using 300 iterations of total epochs as warmup."
            )
            warmup_steps = 300
        else:
            logger.info(f"[SCHEDULER] Warmup epochs are {warmup_steps} iterations.")

        scheduler = get_wsd_scheduler(
            optimizer,
            lr=opt_cfg.lr,
            warmup_end=warmup_steps,
            standard_end=None,
            decay_end=None,
            warmup_start_lr=scheduler_cfg["warmup_start_lr"],
            min_lr=None,
            phase="train",
            decay_function="linear",
        )
    else:
        raise NotImplementedError(f"Scheduler {scheduler_cfg['name']} not implemented.")

    # Set more parameters
    scheduler.num_steps = total_grad_steps

    return scheduler

def load_scheduler(load_cfg):
    """
    Load scheduler state from a checkpoint.
    """
    path = os.path.join(load_cfg.path, "scheduler.pth")

    # Get scheduler state
    if not os.path.exists(path):
        raise ValueError(f"Scheduler state not found at {path}")

    return torch.load(f=path, map_location="cpu")


def setup_optimizer(model, opt_cfg):
    # load_cfg = cfg.LoadConfig
    # opt_cfg = cfg.OptimizerConfig

    # Optimizer
    optimizer_cls = getattr(torch.optim, opt_cfg.optimizer)

    optimizer = optimizer_cls(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=opt_cfg.lr,
        betas=tuple(opt_cfg.betas),
        eps=opt_cfg.eps,
        weight_decay=opt_cfg.weight_decay,
    )

    logger.info(f"Optimizer configuration: {optimizer}")

    return optimizer


def setup_gradients(model, opt_cfg):
    """
    Set gradients to zero.
    """
    
    black_list = opt_cfg.optimize_weights.black_list
    white_list = opt_cfg.optimize_weights.white_list
    assert black_list is None or white_list is None, "Cannot have both black_list and white_list"

    if white_list is not None:
        for name, param in model.named_parameters():
            if any([p in name for p in white_list]):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
    elif black_list is not None:
        for name, param in model.named_parameters():
            if any([p in name for p in black_list]):
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
    else:
        for param in model.parameters():
            param.requires_grad_(True)


    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Setting up gradients:\n"
                f"- Trainable parameters: {trainable_params:,} / {total_params:,}"
                )

    logger.debug(
        "Trainable parameters:\n"
        + "\n".join(
            [
                f"{name}: {param.requires_grad}"
                for name, param in model.named_parameters()
                if param.requires_grad
            ]
        )
    )
    logger.debug(
        "Non-trainable parameters:\n"
        + "\n".join(
            [
                f"{name}: {param.requires_grad}"
                for name, param in model.named_parameters()
                if not param.requires_grad
            ]
        )
    )
