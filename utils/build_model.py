import importlib
import os
import torch
from types import SimpleNamespace
from utils.config import Config
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils.utils import is_valid_hf_model_name
from utils.logging import logger
from utils.distributed import local_rank, world_size
import external_models


class LocalHfCacheError(RuntimeError):
    pass


def import_model_registration_modules(module_names):
    """Import optional modules that register custom Transformers model types."""
    imported = []
    for module_name in module_names or []:
        module_name = module_name.strip()
        if not module_name:
            raise ValueError("Model registration module names must not be empty.")
        try:
            imported.append(importlib.import_module(module_name))
        except Exception as exc:
            raise RuntimeError(
                f"Could not import model registration module {module_name!r}."
            ) from exc
    return imported


def resolve_hf_model_source(model_id: str, local_files_only: bool = False) -> str:
    if not local_files_only or os.path.isdir(model_id):
        return model_id

    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_id, local_files_only=True)
    except Exception as exc:
        raise LocalHfCacheError(
            f"Could not resolve {model_id!r} from the local Hugging Face cache. "
            "Populate the cache first or rerun without local_files_only."
        ) from exc


def build_local_model(builder_cfg=None, dtype=None, device=None, **kwargs):
    from components.registry import Registry

    if type(builder_cfg) == str:
        assert os.path.exists(builder_cfg), f"Path does not exist: {builder_cfg}"
        builder_cfg = Config.from_json(builder_cfg)
    
    assert isinstance(builder_cfg, Config), "builder_cfg must be a Config object"

    model = Registry(builder_cfg.name)(
        config=builder_cfg,
        **builder_cfg.input,
        device=device, 
        dtype=dtype,
        **kwargs
        )

    return model


def build_hf_model(
    model_id: str,
    attn_implementation: str = "flash_attention_2",
    dtype: torch.dtype = None,
    **kwargs
):
    """
    Build a Hugging Face model.
    """
    config = AutoConfig.from_pretrained(model_id)

    if isinstance(config, external_models.LlamaConfig):
        get_model = external_models.LlamaForCausalLM._from_config
    elif isinstance(config, external_models.Qwen2Config):
        get_model = external_models.Qwen2ForCausalLM._from_config
    elif isinstance(config, external_models.FalconMambaConfig):
        get_model = external_models.FalconMambaForCausalLM._from_config
    else:
        get_model = AutoModelForCausalLM.from_config

    try:
        model = get_model(config, attn_implementation=attn_implementation)
    except Exception as e:
        if "flash_attention" in str(e):
            logger.info(f"{model_id} does not support Flash Attention. Using default attention.")
        model = get_model(config, attn_implementation="eager")

    return model.to(dtype)


def load_hf_model(
    model,
    model_id: str,
    access_token: str = None,
    device=None,
    **kwargs
):
    """
    Load a Hugging Face model.
    """
    model = model.from_pretrained(
        pretrained_model_name_or_path=model_id,
        token=access_token,
        device_map=device,
        cache_dir=os.environ["HF_HUB_CACHE"],
        **kwargs
    )

    return model


def build_model(
        # dir, # str or Config
        dir: str = None,
        components_cfg: Config = None,
        attn_implementation: str = "flash_attention_2", 
        mixed_precision=False,
        model_dtype=torch.bfloat16, 
        device='cpu',
        **kwargs
        ):
    """
    Load model from dir

    Args:
        dir: directory of the model
    Returns:
        model: the model
        tokenizer: the tokenizer
    """
    assert (dir is None) != (components_cfg is None), "Either dir or components_cfg must be provided, and not both"
    assert isinstance(dir, str) or isinstance(components_cfg, Config), "dir must be a string or components_cfg must be a Config object"
    model_dtype = getattr(torch, model_dtype) if isinstance(model_dtype, str) else model_dtype
    if "flash" in attn_implementation and model_dtype != torch.bfloat16 and not mixed_precision:
        logger.warning("Flash Attention is only supported with bfloat16. Turning off Flash Attention.")
        attn_implementation = "eager"

    
    if dir is not None and is_valid_hf_model_name(dir):
        return build_hf_model(model_id=dir, attn_implementation=attn_implementation, dtype=model_dtype, device=device)

    if dir is not None and os.path.exists(dir):
        loaded_cfg = Config.from_json(os.path.join(dir, "config.json"))
        components_cfg = (
            loaded_cfg.ComponentsConfig
            if hasattr(loaded_cfg, "ComponentsConfig")
            else loaded_cfg
        )
        dir = None
    
    if components_cfg is not None:
        return build_local_model(builder_cfg=components_cfg, dtype=model_dtype, device=device, attn_implementation=attn_implementation)

    raise ValueError("Model not found")


def build_n_load_model(
    model_id: str,
    attn_implementation: str = "eager",
    model_dtype: torch.dtype = torch.float32,
    device=None,
    access_token: str = None,
    **kwargs,
):
    """
    Build a model and load weights for generation.

    Returns a small object with a ``.model`` attribute for backwards
    compatibility with generation/generate.py.
    """
    if os.path.isdir(model_id):
        config_path = os.path.join(model_id, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Local model directory is missing config.json: {model_id}")
        loaded_cfg = Config.from_json(config_path)
        if model_dtype in {None, "auto"}:
            model_dtype = _get_nested_value(loaded_cfg, "TrainConfig", "model_dtype")
        model_dtype = _resolve_model_dtype(model_dtype)
        components_cfg = (
            loaded_cfg.ComponentsConfig
            if hasattr(loaded_cfg, "ComponentsConfig")
            else loaded_cfg
        )
        model = build_local_model(
            builder_cfg=components_cfg,
            dtype=model_dtype,
            device=device,
            attn_implementation=attn_implementation,
            **kwargs,
        )
        from utils.weights_utils import load_weights

        model, _, _ = load_weights(
            model,
            load_cfgs=[Config.from_dict({"path": model_id})],
            model_dtype=model_dtype,
        )
        return SimpleNamespace(model=model)

    model_dtype = torch.float32 if model_dtype in {None, "auto"} else _resolve_model_dtype(model_dtype)
    if is_valid_hf_model_name(model_id):
        model_source = resolve_hf_model_source(
            model_id,
            local_files_only=kwargs.get("local_files_only", False),
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_source,
            token=access_token or os.environ.get("HF_TOKEN") or None,
            cache_dir=os.environ.get("HF_HUB_CACHE", None),
            torch_dtype=model_dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )
        return SimpleNamespace(model=model)

    raise ValueError(
        f"Model '{model_id}' is neither an existing local checkpoint directory nor a valid Hugging Face model ID."
    )


def _resolve_model_dtype(model_dtype):
    if model_dtype in {None, "auto"}:
        return torch.float32
    return getattr(torch, model_dtype) if isinstance(model_dtype, str) else model_dtype


def get_tokenizer(model_id, **kwargs):
    model_source = resolve_hf_model_source(
        model_id,
        local_files_only=kwargs.get("local_files_only", False),
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, **kwargs)
    except Exception as exc:
        tokenizer_source = _tokenizer_source_from_mohawk_checkpoint(model_source)
        if tokenizer_source is None:
            raise ValueError(f"Tokenizer not found for {model_id}.") from exc
        tokenizer_kwargs = dict(kwargs)
        if _checkpoint_prefers_local_tokenizer(model_source):
            tokenizer_kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)

    return tokenizer


def _get_nested_value(obj, *keys):
    for key in keys:
        if not hasattr(obj, key):
            return None
        obj = getattr(obj, key)
    return obj


def _iter_local_configs(path):
    for filename, loader in [
        ("config.json", Config.from_json),
        ("config.yaml", Config.from_yaml),
        ("config.yml", Config.from_yaml),
    ]:
        config_path = os.path.join(path, filename)
        if os.path.exists(config_path):
            try:
                yield loader(config_path)
            except Exception:
                continue


def _tokenizer_source_from_mohawk_checkpoint(path):
    if not os.path.isdir(path):
        return None
    for cfg in _iter_local_configs(path):
        for keys in [
            ("TrainConfig", "tokenizer"),
            ("TrainDataConfig", "Tokenize", "tokenizer"),
            ("TeacherConfig", "tokenizer"),
        ]:
            tokenizer_source = _get_nested_value(cfg, *keys)
            if tokenizer_source:
                return tokenizer_source
    return None


def _checkpoint_prefers_local_tokenizer(path):
    if not os.path.isdir(path):
        return False
    for cfg in _iter_local_configs(path):
        local_files_only = _get_nested_value(
            cfg, "TrainDataConfig", "Tokenize", "local_files_only"
        )
        if local_files_only is not None:
            return bool(local_files_only)
    return False
