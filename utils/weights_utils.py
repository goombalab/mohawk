import json
import os
import torch

from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoConfig
from utils.utils import is_valid_hf_model_name, recurrent_fn
from utils.logging import logger
from utils.config import Config
from utils.init_weights import init_ssm_from_attention_layer


def get_model_config(path):
    """
    Load model config from a checkpoint path.
    
    Args:
        path: Path to model (local directory or HuggingFace model ID)
        
    Returns:
        Config object (or None if config not found)
    """
    is_hf_model = is_valid_hf_model_name(path)
    
    if is_hf_model:
        try:
            config = AutoConfig.from_pretrained(
                path,
                cache_dir=os.environ.get("HF_HUB_CACHE", None),
            )
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from HuggingFace model {path}: {e}")
            return None
    elif os.path.exists(os.path.join(path, "config.json")):
        try:
            with open(os.path.join(path, "config.json"), "r") as f:
                config_dict = json.load(f)
            # Create a simple object to access config attributes
            class SimpleConfig:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)
            return SimpleConfig(config_dict)
        except Exception as e:
            logger.warning(f"Failed to load config.json from {path}: {e}")
            return None
    else:
        return None


def get_model_weights(path, model_dtype=torch.float32):
    assert path is not None, "Path is None."
    is_hf_model = is_valid_hf_model_name(path)

    # Get model weights
    if os.path.exists(os.path.join(path, "model.safetensors")):
        model_weights = load_file(
            os.path.join(path, "model.safetensors"),
            device="cpu"
            )
    elif os.path.exists(os.path.join(path, "pytorch_model.bin")):
        logger.warning(f"Binary model weights found at {path}, consider using SafeTensors.")
        model_weights = torch.load(
            f=os.path.join(path, "pytorch_model.bin"), 
            map_location="cpu"
        )
    elif is_hf_model:
        model_weights = (
            AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=path,
                device_map="cpu",
                cache_dir=os.environ.get("HF_HUB_CACHE", os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub")),
                torch_dtype=model_dtype,
                offload_folder=os.environ.get("MODEL_OFFLOAD_DIR", os.path.join(os.path.expanduser("~"), ".cache/mohawk/offload")),  # Offload unused weights to disk if necessary
                offload_state_dict=True,                  # Offload CPU weights as needed
            ).state_dict()
        )
    else:
        raise ValueError(f"Model weights not found at {path}")

    return model_weights

def filter_keys(
    model_weights,
    black_list=[],
    white_list=None,
    rename={},
):
    """
    Filter model keys.
    """

    # 1. rename keys
    if rename is not None:
        for old, new in rename.items():
            model_weights = {k.replace(old, new): v for k, v in model_weights.items()}

    # 2. black_list (remove keys that contain the black_listed strings)
    if black_list is not None:
        model_weights = {
            k: v
            for k, v in model_weights.items()
            if not any([s in k for s in black_list])
        }

    # 3. white_list (remove keys that do not contain the white_listed strings):
    if white_list is not None:
        model_weights = {
            k: v for k, v in model_weights.items() if any([s in k for s in white_list])
        }

    return model_weights


def _load_ssm_from_attention(model, item, model_dtype):
    """
    Initialize SSM layers from teacher attention weights (modifies model in-place).
    Uses the same weight loading mechanism as standard loading.
    
    Args:
        model: Student model to initialize (modified in-place)
        item: Load config item with path, strategy
        model_dtype: Model dtype
    
    Returns:
        tuple: (loaded_keys, missing_keys) - Lists of parameter keys that were successfully loaded
               and missing during SSM initialization
    """
    assert item.path is not None, "No path provided for init_ssm_from_attention."
    logger.info(f"Initializing SSM from attention in {item.path}")
    try:
        # Load teacher config to get head_dim
        teacher_config = get_model_config(item.path)
        
        # Extract head_dim from config
        head_dim = None
        if teacher_config is not None:
            # Try different common attribute names for head_dim
            head_dim = getattr(teacher_config, "head_dim", None)
            if head_dim is None:
                # Calculate from hidden_size and num_attention_heads if available
                hidden_size = getattr(teacher_config, "hidden_size", None)
                num_heads = getattr(teacher_config, "num_attention_heads", None)
                if hidden_size is not None and num_heads is not None:
                    head_dim = hidden_size // num_heads
        

        assert head_dim is not None, f"Could not determine head_dim from config at {item.path}. \
            \n Please ensure the model has a valid config.json or is a HuggingFace model."
        
        # Load teacher weights (same as standard loading)
        teacher_weights = get_model_weights(item.path, model_dtype)
        
        # Get student model layers
        if hasattr(model, 'backbone'):
            student_layers = model.backbone.layers
            layer_prefix = "backbone.layers"
        elif hasattr(model, 'model'):
            student_layers = model.model.layers
            layer_prefix = "model.layers"
        else:
            raise ValueError("Could not find layers in student model (expected 'backbone' or 'model')")
        
        # Track which layers succeeded and failed
        transferred_layers = []
        not_transferred_layers = []
        all_loaded_keys = []
        
        # Extract attention weights and initialize SSM for each layer
        for layer_idx, student_layer in enumerate(student_layers):
            # Check if student has SSM (mixer)
            if not hasattr(student_layer, 'mixer'):
                not_transferred_layers.append(layer_idx)
                continue
            
            # Try to find attention weights in teacher state dict
            # Common patterns: "model.layers.{layer_idx}.self_attn.*" or "layers.{layer_idx}.self_attn.*"
            attn_prefixes = [
                f"model.layers.{layer_idx}.self_attn",
                f"layers.{layer_idx}.self_attn",
                f"backbone.layers.{layer_idx}.self_attn",
            ]
            
            attn_weights = {}
            attn_prefix = None
            for prefix in attn_prefixes:
                if any(k.startswith(prefix) for k in teacher_weights.keys()):
                    attn_prefix = prefix
                    break
            
            if attn_prefix is None:
                not_transferred_layers.append(layer_idx)
                continue
            
            # Extract attention weights
            for key, value in teacher_weights.items():
                if key.startswith(attn_prefix):
                    # Remove prefix to get the weight name (e.g., "q_proj.weight")
                    weight_name = key[len(attn_prefix) + 1:]  # +1 for the "."
                    attn_weights[weight_name] = value
            
            if not attn_weights:
                not_transferred_layers.append(layer_idx)
                continue
            
            # Initialize SSM from attention weights directly
            if "q_proj.weight" in attn_weights:
                try:
                    stats = init_ssm_from_attention_layer(
                        student_ssm_layer=student_layer.mixer,
                        teacher_attn_weights=attn_weights,
                        head_dim=head_dim,
                    )
                    transferred_layers.append(layer_idx)
                    # Collect loaded keys with full path: e.g., "backbone.layers.0.mixer.q_proj.weight"
                    for param_name in stats.get("loaded_keys", []):
                        full_key = f"{layer_prefix}.{layer_idx}.mixer.{param_name}"
                        all_loaded_keys.append(full_key)
                except Exception as e:
                    not_transferred_layers.append(layer_idx)
                    continue
        
        # Log results
        logger.info(
            f"Succesfully loaded QKV from {item.path} into SSM layers: \
            \n - Loaded layers: {len(transferred_layers)} \
            \n - Skipped layers: {len(not_transferred_layers)} \
            \n - Loaded keys: {len(all_loaded_keys)}"
        )
        return all_loaded_keys, []
    except Exception as e:
        logger.error(f"Failed to initialize SSM from attention: {e}")
        raise


def _load_standard_weights(model, item, model_dtype):
    """
    Load standard weights from a checkpoint (modifies model in-place).
    
    Args:
        model: Model to load weights into (modified in-place)
        item: Load config item with path, black_list, white_list, rename, etc.
        model_dtype: Model dtype
    
    Returns:
        tuple: (loaded_keys, missing_keys) - Lists of keys that were loaded and missing
    """
    if item.path is None:
        return [], []

    # Get model weights & filter keys
    model_weights = filter_keys(
        model_weights=get_model_weights(item.path, model_dtype),
        black_list=item.get("black_list", []), 
        white_list=item.get("white_list", None),
        rename=item.get("rename", {}),
    )

    # Load model weights (strictness enforced manually below)
    result = model.to("cpu").load_state_dict(model_weights, strict=False)
    loaded_keys = {
        k: v for k, v in model.state_dict().items() if k not in result.missing_keys
    }
    all_loaded_keys = list(loaded_keys.keys())
    all_missing_keys = list(result.missing_keys)
    total_model_keys = len(model.state_dict())

    # Log results
    logger.info(
        f"Succesfully loaded from {item.path}: \
        \n - Loaded keys: {len(loaded_keys)} \
        \n - Missing keys: {len(result.missing_keys)} \
        \n - Unexpected keys: {len(result.unexpected_keys)} \
        \n - Total model keys: {total_model_keys}"
    )
    logger.debug(
        f"Successfully loaded the following keys from {item.path}:\n"
        + "\n".join(list(loaded_keys.keys()))
    )
    logger.debug(
        f"MISSING KEYS: Could not load the following keys from {item.path}\n"
        + "\n".join(result.missing_keys)
    )
    logger.debug(
        f"UNEXPECTED KEYS: The following keys were not expected from {item.path}\n"
        + "\n".join(result.unexpected_keys)
    )

    # Validate strictness
    assert (
        len(result.missing_keys) == 0 or item.get("allow_missing_keys", True)
    ), f"Missing keys: {result.missing_keys}"
    assert (
        len(result.unexpected_keys) == 0 or item.get("allow_unexpected_keys", True)
    ), f"Unexpected keys: {result.unexpected_keys}"

    # Clean up
    del model_weights
    del result
    
    return all_loaded_keys, all_missing_keys


def load_weights(
        model, 
        model_dtype, 
        load_cfgs=[]
        ):
    """
    Load weights from a checkpoint (modifies model in-place).
    
    Supports different loaders via the 'type' field:
    - 'weights_loader' (default): Standard weight loading
    - 'init_ssm_from_attention': Initialize SSM layers from teacher attention
    
    Args:
        model: Model to load weights into (modified in-place)
        model_dtype: Model dtype
        load_cfgs: List of load config items
    
    Returns:
        tuple: (model, loaded_keys, missing_keys)
            - model: The model with loaded weights (modified in-place)
            - loaded_keys: List of all keys that were successfully loaded (aggregated across all load configs)
            - missing_keys: List of all keys that were missing (aggregated across all load configs)
    """
    all_loaded_keys = []
    all_missing_keys = []
    
    for item in load_cfgs:
        assert isinstance(item, dict) or isinstance(item, Config), "item must be a dict or Config."

        # Get loader type (default to weights_loader for backward compatibility)
        loader_type = item.get("type", "weights_loader")
        
        # Skip if path is None (unless it's init_ssm_from_attention which uses path differently)
        if item.path is None and loader_type != "init_ssm_from_attention":
            continue

        # Handle different loader types
        if loader_type == "init_ssm_from_attention":
            loaded_keys, missing_keys = _load_ssm_from_attention(model, item, model_dtype)
            all_loaded_keys.extend(loaded_keys)
            all_missing_keys.extend(missing_keys)
        
        elif loader_type == "weights_loader" or loader_type is None:
            loaded_keys, missing_keys = _load_standard_weights(model, item, model_dtype)
            all_loaded_keys.extend(loaded_keys)
            all_missing_keys.extend(missing_keys)
        
        else:
            raise ValueError(f"Unknown loader type: {loader_type}. Supported: 'weights_loader', 'init_ssm_from_attention'")

    # Deduplicate and clean up keys
    all_loaded_keys = list(set(all_loaded_keys))  # Remove duplicates
    all_missing_keys = list(set(all_missing_keys))  # Remove duplicates
    all_missing_keys = [k for k in all_missing_keys if k not in all_loaded_keys]  # Remove loaded keys from missing
    
    n_keys = len(all_loaded_keys)
    logger.info(f"Loaded {n_keys} keys in total.")
    
    return model, all_loaded_keys, all_missing_keys

@recurrent_fn(n_tries=3)
def save_weights(model_state, tmp_dir, save_dir, opt_state=None, scheduler_state=None, config=None, safe_tensors=True):
    # Save weights at tmp_dir
    filename = "model.safetensors" if safe_tensors else "pytorch_model.bin"
    _save_dict(model_state, tmp_dir, save_dir, filename, safe_tensors)

    # Save the optimizer state
    _save_dict(opt_state, tmp_dir, save_dir, "optimizer.pth", False)
    
    # Save scheduler state
    _save_dict(scheduler_state, tmp_dir, save_dir, "scheduler.pth", False)

    # Save the config at tmp_dir
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        f.write(json.dumps(config.to_dict()).replace("\\n", "\n"))

    logger.info(f"[TRAINING_WRAPPER] Checkpoint saved successfully at {save_dir}")

def _save_dict(state_dict, tmp_dir, save_dir, filename, safe_tensors=True):
    def move_to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif isinstance(obj, dict):
            return {k: move_to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [move_to_cpu(v) for v in obj]
        else:
            return obj

    if state_dict is None:
        return None
    assert isinstance(state_dict, dict), "State must be a dictionary."
    if safe_tensors:
        assert all(isinstance(v, (torch.Tensor)) for v in state_dict.values()), \
            "SafeTensors requires all values to be torch.Tensor (flat dictionary)."

    os.makedirs(tmp_dir, exist_ok=True)
    save_func = save_file if safe_tensors else torch.save
    state_dict = move_to_cpu(state_dict)
    save_func(state_dict, os.path.join(tmp_dir, filename))

    # Move to save_dir
    os.makedirs(save_dir, exist_ok=True)
    os.system(f"mv -f {tmp_dir}/* {save_dir}")
    return dir
