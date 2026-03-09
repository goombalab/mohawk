import os
import torch
from utils.config import Config
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils.utils import is_valid_hf_model_name
from utils.logging import logger
from utils.distributed import local_rank, world_size
import external_models

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
        components_cfg = Config.from_json(dir + "/config.json")
        dir = None
    
    if components_cfg is not None:
        return build_local_model(builder_cfg=components_cfg, dtype=model_dtype, device=device, attn_implementation=attn_implementation)

    raise ValueError("Model not found")


def get_tokenizer(model_id):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except:
        raise ValueError(f"Tokenizer not found for {model_id}.")    

    return tokenizer
