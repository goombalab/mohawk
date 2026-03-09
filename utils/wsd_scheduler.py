from torch.optim.lr_scheduler import LambdaLR


# Custom scheduler function with parameters
def lr_wsd_lambda(
    step,
    max_lr,
    min_lr,
    warmup_start_lr,
    warmup_end,
    standard_end,
    decay_end,
    decay_function,
):
    if step < warmup_end:
        return (
            step / warmup_end * (max_lr - warmup_start_lr) + warmup_start_lr
        ) / max_lr
    elif step < standard_end:
        return 1
    elif step < decay_end:
        return (
            decay_function(
                step - standard_end, decay_end - standard_end, max_lr - min_lr
            )
            + min_lr
        ) / max_lr
    else:
        return min_lr / max_lr


def lr_wsd_training_only_lambda(step, max_lr, warmup_start_lr, warmup_end):
    if step < warmup_end:
        return (
            step / warmup_end * (max_lr - warmup_start_lr) + warmup_start_lr
        ) / max_lr
    else:
        return 1


def lr_wsd_decay_only_lambda(step, max_lr, min_lr, decay_steps, decay_function):
    if step < decay_steps:
        return (decay_function(step, decay_steps, max_lr - min_lr) + min_lr) / max_lr
    else:
        return min_lr / max_lr


def linear_decay_function(step, decay_steps, lr_difference):
    return (decay_steps - step) / decay_steps * lr_difference


def get_wsd_scheduler(
    optimizer,
    lr,
    warmup_end,
    standard_end,
    decay_end,
    warmup_start_lr=0.0,
    min_lr=0.0,
    phase="entire",
    decay_function="linear",
):
    """
    WSD (Warmup-Stable-Decay) Scheduler.

    Parameters:
    optimizer (torch.optim.Optimizer): Optimizer.
    lr (float): Learning rate.
    warmup_end (int): End of warmup phase.
    standard_end (int): End of stable phase.
    decay_end (int): End of decay phase.
    warmup_start_lr (float): Start of warmup phase.
    min_lr (float): Minimum learning rate.
    phase (str): Phase of the scheduler. Options: "entire" (warmup-stable-decay), "train" (warmup-stable), "decay" (decay).
    decay_function (str): Decay function. Options: "linear".

    Returns:
    LambdaLR: Scheduler.

    """
    decay_fn = None
    if decay_function == "linear":
        decay_fn = linear_decay_function

    if decay_fn is None:
        raise Exception("Not a valid decay function.")

    lr_lambda = None
    if phase == "entire":
        lr_lambda = lambda step: lr_wsd_lambda(
            step,
            lr,
            min_lr,
            warmup_start_lr,
            warmup_end,
            standard_end,
            decay_end,
            decay_fn,
        )
    elif phase == "train":
        lr_lambda = lambda step: lr_wsd_training_only_lambda(
            step, lr, warmup_start_lr, warmup_end
        )
    elif phase == "decay":
        lr_lambda = lambda step: lr_wsd_decay_only_lambda(
            step, lr, min_lr, decay_end - standard_end, decay_fn
        )

    if lr_lambda is None:
        raise Exception("Not a valid phase.")

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler
