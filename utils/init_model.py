import torch
import sys
from contextlib import contextmanager

from utils.logging import logger
from utils.build_model import build_model
from utils.weights_utils import load_weights
from utils.distributed import local_rank, barrier, world_size
from training_wrapper import CentralizedTrainingWrapper, FSDPTrainingWrapper, DDPTrainingWrapper
from utils.config import Config


def move_meta_to_cpu(model: torch.nn.Module, params_map = {}):
    """
    Traverse all parameters and buffers in a model.
    Move tensors stored on 'meta' to 'cpu', handling submodules correctly.
    """
    # Traverse all parameters in a model
    for name, param in model.named_parameters(recurse=False):  # Only direct parameters
        if param.device.type != 'meta':
            continue
        new_param = torch.empty_like(param, device='cpu', requires_grad=False)
        model.register_parameter(name, torch.nn.Parameter(new_param))
        
    
    # Traverse all buffers in a model
    for name, buffer in model.named_buffers(recurse=False):  # Only direct buffers
        if buffer.device.type == 'meta':
            with torch.no_grad():
                new_buffer = torch.empty_like(buffer, device='cpu')
            if isinstance(new_buffer, torch.nn.Parameter):
                new_buffer = new_buffer.detach()
            is_persistent = name not in model._non_persistent_buffers_set
            model.register_buffer(name, new_buffer, persistent=is_persistent)

    # Recursively handle submodules
    for child_name, child_module in model.named_children():
        move_meta_to_cpu(child_module, params_map)


@contextmanager
def parameter_device(device: torch.device):
    """
    A context manager to globally override torch.nn.Parameter to ensure
    all parameter initializations (even from direct imports) respect the device.
    
    Args:
        device (torch.device): The target device for parameter initialization.
    """
    # Save the original Parameter class
    original_parameter = torch.nn.parameter.Parameter

    # Create a new class that overrides Parameter behavior
    class DeviceParameter(original_parameter):
        def __new__(cls, data, requires_grad=True):
            data = data.to(device)
            return super().__new__(cls, data, requires_grad=requires_grad)

    try:
        # Replace the Parameter in the torch.nn.parameter module
        torch.nn.parameter.Parameter = DeviceParameter
        # Also patch it in sys.modules to handle all imports
        modules_snapshot = list(sys.modules.items())  # Take a snapshot of sys.modules
        for module_name, module in modules_snapshot:
            if hasattr(module, "Parameter") and module.Parameter is original_parameter:
                setattr(module, "Parameter", DeviceParameter)

        yield
    finally:
        # Restore the original Parameter in all locations
        torch.nn.parameter.Parameter = original_parameter
        modules_snapshot = list(sys.modules.items())  # Snapshot again for cleanup
        for module_name, module in modules_snapshot:
            if hasattr(module, "Parameter") and module.Parameter is DeviceParameter:
                setattr(module, "Parameter", original_parameter)

def get_init_fn(init_fn):
    if init_fn == "lazy":
        return lazy_init
    elif init_fn == "eager":
        return eager_init
    else:
        raise ValueError(f"init_fn must be 'lazy' or 'eager', got {init_fn}")

def _get_wrapper_class(details_cfg):
    """Get the appropriate wrapper class based on world_size and config."""
    if world_size == 1:
        return CentralizedTrainingWrapper
        
    wrapper_type = details_cfg.get("wrapper_type", "fsdp").lower()
    if wrapper_type == "ddp":
        return DDPTrainingWrapper
    elif wrapper_type == "fsdp":
        return FSDPTrainingWrapper
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}. Supported: 'ddp', 'fsdp'")

    
def eager_init(
        cfg,
        details_cfg,
        load_cfg,
        mode,
        components_cfg : Config = None,
    ):
        """
        Initialize all model weights and then load weights to model - across all ranks.
        """
        model_dtype = getattr(torch, details_cfg.model_dtype)
        assert isinstance(model_dtype, torch.dtype), f"model_dtype must be a torch.dtype, got {model_dtype}"

        logger.info("[TRAINING_WRAPPER] Initializing model...")
        wrapper_cls = _get_wrapper_class(details_cfg)

        model = build_model(components_cfg=components_cfg, **details_cfg)
        if hasattr(load_cfg, "model"):
            model, _, _ = load_weights(model, load_cfgs=load_cfg.model, model_dtype=model_dtype)
        else:
            logger.warning("[TRAINING_WRAPPER] No weights to load in eager initialization.")
            
        return wrapper_cls(model=model, config=cfg, mode=mode, **details_cfg)

def lazy_init(
        cfg,
        details_cfg,
        load_cfg,
        mode,
        components_cfg : Config = None,
):
    """
    Skip model initialization with random weights and load weights only on rank 0.
    """
    model_dtype = getattr(torch, details_cfg.model_dtype)
    assert isinstance(model_dtype, torch.dtype), f"model_dtype must be a torch.dtype, got {model_dtype}"

    logger.info(f"[TRAINING_WRAPPER] Lazy initialization of model on rank {local_rank}...")

    with parameter_device(torch.device("meta")): # prev: with torch.device("meta"):
        model = build_model(components_cfg=components_cfg, **details_cfg)

    # Load weights
    if local_rank == 0:
        move_meta_to_cpu(model) # Move meta tensors to CPU
        model, loaded_keys, missing_keys = load_weights(model, load_cfgs=load_cfg.model, model_dtype=model_dtype)
        assert len(missing_keys) == 0, f"Cannot miss any keys during lazy initialization"

    # Load optimizer state
    if local_rank == 0:
        pass # todo
    
    barrier() # Wait for all processes to load weights
    
    # Choose wrapper class
    wrapper_cls = _get_wrapper_class(details_cfg)
    
    # Initialize other ranks' models if needed (for weights_from_rank0)
    if world_size > 1 and local_rank != 0:
        move_meta_to_cpu(model)
    
    # Create wrapper - it will handle weights_from_rank0 if needed
    if world_size > 1:
        wrapper = wrapper_cls(model=model, config=cfg, mode=mode, weights_from_rank0=True, **details_cfg)
    else:
        wrapper = wrapper_cls(model=model, config=cfg, mode=mode, **details_cfg)
    
    barrier()
    return wrapper
