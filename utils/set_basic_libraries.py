EXECUTED = {}

def set_temporary_folder():
    """
    Sets the temporary folder to /tmp/<username>/<job_id> where job_id 
    is the SLURM job id if available, else 0.
    """
    import os
    job_id = os.environ.get("SLURM_JOBID", "0")
    username = os.getenv("USER") or os.getenv("USERNAME")
    os.environ["TMPDIR"] = (os.getenv("TMPDIR") or "/tmp") + f"/{username}/{job_id}"
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    EXECUTED["set_temporary_folder"] = True

def suppress_warnings():
    """
    Suppresses FutureWarning and UserWarnings.

    Note: order of imports matters.
    This function should be called before any other imports,
    but after setting the temporary folder.
    """
    assert EXECUTED.get("set_temporary_folder"), "set_temporary_folder() must be called before suppress_warnings()"

    # --- Python warnings ---
    import warnings
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", UserWarning)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.showwarning = lambda *args, **kwargs: None
        # Import libs whose first-import warnings you want to silence
        import numpy
        from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
        from pl_bolts.utils.stability import UnderReviewWarning  # noqa: F401

    # --- Python logging (root) ---
    import logging as pylog
    pylog.getLogger().setLevel(pylog.ERROR)

    # --- Option A: quiet HuggingFace Datasets & Hub loggers ---
    from datasets.utils.logging import set_verbosity as ds_set_verbosity
    from huggingface_hub.utils import logging as hub_logging
    ds_set_verbosity(pylog.ERROR)
    hub_logging.set_verbosity(pylog.ERROR)

    # --- Transformers logger + progress bars ---
    from transformers.utils import logging as t_logging
    from functools import partial
    t_logging.set_verbosity_error()
    # disable tqdm bars exposed via transformers' logging module
    t_logging.tqdm = partial(t_logging.tqdm, disable=True)

    # --- Datasets progress bars ---
    import datasets
    datasets.logging.disable_progress_bar()

    EXECUTED["suppress_warnings"] = True


def print_stacktrace_on_timeout():
    """
    1. Enables faulthandler to dump stack traces on timeout. 
    2. Registers a signal handler to trigger the timeout handler when a signal is received.
    """
    import faulthandler
    import signal
    import sys
    faulthandler.enable()
    def timeout_handler(signum, frame):
        print(f"Signal {signum} received! Dumping stack traces...")
        faulthandler.dump_traceback(file=sys.stdout)
    signal.signal(signal.SIGUSR1, timeout_handler)
    EXECUTED["print_stacktrace_on_timeout"] = True

def set_formats():
    import pandas as pd
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 100)
    EXECUTED["set_formats"] = True


set_temporary_folder()
suppress_warnings()
print_stacktrace_on_timeout()
set_formats()
