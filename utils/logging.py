import os
import contextlib
from rich.console import Console

__all__ = ["wandb", "logger"]

class DummyObject:
    def __init__(self, *args, **kwargs):
        return None

    def __getattr__(self, *args, **kwargs):
        return DummyObject()

    def __call__(self, *args, **kwargs):
        return DummyObject()

    def __str__(self):
        return "DummyObject"

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        """Return False to indicate that the exception was not handled."""
        return False 

class LoggerProxy:
    _instance = None
    _log_levels = {"trace", "debug", "info", "success", "warning", "error", "critical"}
    _logged_messages = {level: set() for level in _log_levels}

    def __new__(cls, *args, **kwargs):
        """
        This is a singleton class.
        """
        if cls._instance is None:
            cls._instance = super(LoggerProxy, cls).__new__(cls)
        return cls._instance

    def warning(self, message, *args, once=False, **kwargs):
        if once and message in self._logged_messages["warning"]:
            return
        if once:
            self._logged_messages["warning"].add(message)
        assert _state["loguru"] is not None, "Logger is not initialized."
        return _state["loguru"].warning(message, *args, **kwargs)
    
    def info(self, message, *args, once=False, **kwargs):
        if once and message in self._logged_messages["info"]:
            return
        if once:
            self._logged_messages["info"].add(message)
        assert _state["loguru"] is not None, "Logger is not initialized."
        return _state["loguru"].info(message, *args, **kwargs)
            
    def __getattr__(self, name):
        # Re-fetch the logger every time an attribute (e.g., info, debug) is accessed
        assert _state["loguru"] is not None, "Logger is not initialized."
        return getattr(_state["loguru"], name)  # Return the requested method from the logger


class WandbProxy:
    def __getattr__(self, name):
        if _state["wandb"] is None:
            raise ValueError("Wandb is not initialized.")
        return getattr(_state["wandb"], name)
    

###############################
# INITIALIZE GLOBAL VARIABLES #
###############################
is_master = os.environ.get("RANK", "0") == "0"
_state = {"loguru": None, "wandb": None}
logger =  LoggerProxy() if is_master else DummyObject()
wandb = WandbProxy() if is_master else DummyObject()


########################
#### INIT FUNCTIONS ####
########################
def init_wandb(config, wandb_id=None):

    # Import wandb
    import wandb as _wandb

    wandb_cfg = config.ManagementConfig.wandb
    # Get W&B API key from config or environment variable
    wandb_key = wandb_cfg.get("key") or os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        raise ValueError(
            "W&B API key not found. Please set it in configs/management.yaml "
            "or via the WANDB_API_KEY environment variable."
        )
    try:
        _wandb.login(key=wandb_key)
    except Exception as e:
        print(f"Failed to login to wandb: {e}")
        raise

    return _wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        save_code=wandb_cfg.get("save_code", True),
        name=config.DistillConfig.name,
        group=config.DistillConfig.name,
        resume=wandb_id,
        config=config.to_dict(),
    )

def init_loguru():
    wandb_run = _state["wandb"]
    assert wandb_run is not None, "Wandb is not initialized."
    # import loguru
    from loguru import logger as _logger
    # Remove the default stdout handler
    _logger.remove()

    # Add custom stdout & file handlers
    console = Console()
    _logger.add(sink=console.print, format="{message}", level="INFO")  # to console (stdout)
    log_path = os.path.join(str(wandb_run.dir), "main.log")
    _logger.add(sink=log_path, level="TRACE")  # to log file

    return _logger

@contextlib.contextmanager
def init_logging(cfg, wandb_id=None):
    
    # Use an ExitStack to handle multiple context managers
    with contextlib.ExitStack() as stack:
        
        if is_master:
            # Initialize wandb and logger if the process is the master
            _state["wandb"] = stack.enter_context(init_wandb(cfg, wandb_id))
            _state["loguru"] = init_loguru()
            wandb_id = _state["wandb"].id
        else:
            # Otherwise, use a dummy object
            _state["wandb"] = DummyObject()
            _state["loguru"] = DummyObject()
            wandb_id = None

        # Log basic info
        logger.info(cfg)
        
        job_id = os.environ.get("SLURM_JOBID", "0")
        logger.info(
            f"Information:\n"
            f"- slurm_job_id: {job_id},\n"
            f"- wandb_id: {wandb_id}"
        )
        _state["wandb"].log({"slurm_job_id": job_id})

        try:
            yield
        finally:
            pass
