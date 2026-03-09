import math

import torch
import torch.nn as nn
from utils.logging import logger


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def init_ssm_from_attention_layer(
    student_ssm_layer: nn.Module,
    teacher_attn_weights: dict,
    head_dim: int,
) -> dict:
    """
    Initialize a single SSM layer from teacher attention weights.
    Only transfers weights when dimensions match exactly (conservative strategy).
    
    Args:
        student_ssm_layer: The SSM Mixer module (discrete_mamba2.Mixer)
        teacher_attn_weights: Dict of attention weights with keys like:
            - "q_proj.weight": [num_heads * head_dim, d_model]
            - "q_proj.bias": [num_heads * head_dim] (optional)
            - "k_proj.weight": [num_kv_heads * head_dim, d_model]
            - "k_proj.bias": [num_kv_heads * head_dim] (optional)
            - "v_proj.weight": [num_kv_heads * head_dim, d_model]
            - "v_proj.bias": [num_kv_heads * head_dim] (optional)
            - "o_proj.weight": [d_model, num_heads * head_dim]
            - "o_proj.bias": [d_model] (optional)
        head_dim: Head dimension of the teacher attention.
        
    Returns:
        dict with transfer statistics
    """
    # Assert that it's a discrete_mamba2 Mixer
    assert student_ssm_layer.__class__.__module__ == "components.cores.discrete_mamba2", \
        f"Expected discrete_mamba2.Mixer, got {student_ssm_layer.__class__.__module__}.{student_ssm_layer.__class__.__name__}"
    assert student_ssm_layer.__class__.__name__ == "Mixer", \
        f"Expected Mixer class, got {student_ssm_layer.__class__.__name__}"
    
    stats = {
        "transferred": [],
        "skipped": [],
        "initialized_from_scratch": [],
        "ssm_dimensions": {},
        "attn_dimensions": {},
        "loaded_keys": [],  # List of parameter names that were successfully loaded (relative to mixer layer)
    }
    
    # Get SSM dimensions
    ssm_d_model = student_ssm_layer.d_model
    ssm_d_inner = student_ssm_layer.d_inner
    ssm_n_v_heads = student_ssm_layer.n_v_heads
    ssm_n_qk_heads = student_ssm_layer.n_qk_heads
    ssm_d_state = student_ssm_layer.d_state
    
    stats["ssm_dimensions"] = {
        "d_model": ssm_d_model,
        "d_inner": ssm_d_inner,
        "n_v_heads": ssm_n_v_heads,
        "n_qk_heads": ssm_n_qk_heads,
        "d_state": ssm_d_state,
    }
    
    # Extract attention dimensions from weights
    if "q_proj.weight" not in teacher_attn_weights:
        raise ValueError("Missing q_proj.weight in teacher_attn_weights")
    
    q_weight = teacher_attn_weights["q_proj.weight"]
    attn_d_model = q_weight.shape[1]  # in_features
    attn_q_output_dim = q_weight.shape[0]  # out_features = num_heads * head_dim
    
    # Validate head_dim
    attn_head_dim = head_dim
    if attn_q_output_dim % attn_head_dim != 0:
        raise ValueError(
            f"Provided head_dim={head_dim} does not divide evenly into "
            f"q_proj output dimension {attn_q_output_dim}"
        )
    attn_num_heads = attn_q_output_dim // attn_head_dim
    
    # Infer num_kv_heads from k_proj if available
    if "k_proj.weight" in teacher_attn_weights:
        k_weight = teacher_attn_weights["k_proj.weight"]
        attn_k_output_dim = k_weight.shape[0]
        attn_num_kv_heads = attn_k_output_dim // attn_head_dim
    else:
        attn_num_kv_heads = attn_num_heads
    
    stats["attn_dimensions"] = {
        "d_model": attn_d_model,
        "num_heads": attn_num_heads,
        "num_kv_heads": attn_num_kv_heads,
        "head_dim": attn_head_dim,
    }
    
    # ============================================================
    # 1. Output Projection: Direct copy if dimensions match
    # ============================================================
    assert "o_proj.weight" in teacher_attn_weights, "Missing o_proj.weight in teacher_attn_weights"
    o_weight = teacher_attn_weights["o_proj.weight"]
    # out_proj has shape [out_features, in_features] = [d_model, num_heads * head_dim]
    # Need to check both dimensions match
    assert o_weight.shape[0] == ssm_d_model and o_weight.shape[1] == ssm_d_inner, \
        f"out_proj dimension mismatch: expected [{ssm_d_model}, {ssm_d_inner}], got {list(o_weight.shape)}"
    proj_stats = _move_proj(
        teacher_attn_weights, "o_proj",
        student_ssm_layer.out_proj,
        expected_output_dim=ssm_d_model,
    )
    stats["transferred"].extend(proj_stats["transferred"])
    stats["skipped"].extend(proj_stats["skipped"])
    stats["loaded_keys"].extend(proj_stats["loaded_keys"])
    
    # ============================================================
    # 2. Input Projections: Direct mapping using attention-orented naming
    # ============================================================
    # C (contraction) = Q (queries), B (expansion) = K (keys), X (input) = V (values)
    # SSM has separate projections: q_proj, k_proj, v_proj, A_log_proj
    # Attention has: q_proj, k_proj, v_proj
    
    if ssm_d_model == attn_d_model:
        with torch.no_grad():
            # Get Q weights and bias (required, no fallback)
            q_weight = teacher_attn_weights["q_proj.weight"]  # [num_heads * head_dim, d_model]
            q_bias = teacher_attn_weights.get("q_proj.bias", None)  # [num_heads * head_dim] or None
            
            # Calculate repeat factor for GQA expansion
            repeat_factor = attn_num_heads // attn_num_kv_heads if attn_num_kv_heads < attn_num_heads else 1
            
            # q_proj = Q (queries) - contraction matrix (C)
            proj_stats = _move_proj(
                teacher_attn_weights, "q_proj",
                student_ssm_layer.q_proj,
                expected_output_dim=ssm_n_qk_heads * ssm_d_state,
                fallback_weight=q_weight, fallback_bias=q_bias,
                repeat_factor=1,  # Q doesn't need expansion
            )
            stats["transferred"].extend(proj_stats["transferred"])
            stats["skipped"].extend(proj_stats["skipped"])
            stats["loaded_keys"].extend(proj_stats["loaded_keys"])
            
            # k_proj = K (keys) - expansion matrix (B)
            proj_stats = _move_proj(
                teacher_attn_weights, "k_proj",
                student_ssm_layer.k_proj,
                expected_output_dim=ssm_n_qk_heads * ssm_d_state,
                fallback_weight=q_weight, fallback_bias=q_bias,
                repeat_factor=repeat_factor,
            )
            stats["transferred"].extend(proj_stats["transferred"])
            stats["skipped"].extend(proj_stats["skipped"])
            stats["loaded_keys"].extend(proj_stats["loaded_keys"])
            
            # v_proj = V (values) - input sequence (X)
            proj_stats = _move_proj(
                teacher_attn_weights, "v_proj",
                student_ssm_layer.v_proj,
                expected_output_dim=ssm_d_inner,
                fallback_weight=q_weight, fallback_bias=q_bias,
                repeat_factor=repeat_factor,
            )
            stats["transferred"].extend(proj_stats["transferred"])
            stats["skipped"].extend(proj_stats["skipped"])
            stats["loaded_keys"].extend(proj_stats["loaded_keys"])
            
            # A_log_proj: Initialize from scratch (SSM-specific)
            stats["initialized_from_scratch"].append("A_log_proj")
    else:
        stats["skipped"].append("input projections (d_model mismatch)")
    
    # ============================================================
    # 3. SSM-specific parameters: Initialize from scratch
    # ============================================================
    # D parameter (skip connection) - already initialized to ones, keep it
    stats["initialized_from_scratch"].append("D")
    
    # Conv1d: Keep default initialization
    stats["initialized_from_scratch"].append("conv1d")
    
    # Norm: Keep default initialization
    stats["initialized_from_scratch"].append("norm")
    
    return stats


def _move_proj(
    teacher_attn_weights: dict,
    proj_name: str,
    student_proj: nn.Linear,
    expected_output_dim: int,
    fallback_weight: torch.Tensor = None,
    fallback_bias: torch.Tensor = None,
    repeat_factor: int = 1,
) -> dict:
    """
    Move (copy) projection weights and bias from teacher to student.
    Handles fallback, GQA expansion, dimension checking, and copying.
    
    Args:
        teacher_attn_weights: Dict of teacher attention weights
        proj_name: Name of the projection (e.g., "k_proj", "v_proj")
        student_proj: Student projection layer (nn.Linear)
        expected_output_dim: Expected output dimension for dimension check
        fallback_weight: Weight to use if proj_name.weight doesn't exist
        fallback_bias: Bias to use if proj_name.bias doesn't exist
        repeat_factor: Factor to repeat weights/bias for GQA expansion
        
    Returns:
        dict with transfer statistics (keys: "transferred", "skipped")
    """
    stats = {
        "transferred": [],
        "skipped": [],
        "loaded_keys": [],  # List of parameter names that were successfully loaded
    }
    
    weight_key = f"{proj_name}.weight"
    bias_key = f"{proj_name}.bias"
    
    # Get weights and bias (with fallback if needed)
    if weight_key in teacher_attn_weights:
        weight = teacher_attn_weights[weight_key]
        bias = teacher_attn_weights.get(bias_key, None)
    else:
        weight = fallback_weight
        bias = fallback_bias
    
    # Expand for GQA if needed
    if repeat_factor > 1:
        weight = weight.repeat_interleave(repeat_factor, dim=0)
        if bias is not None:
            bias = bias.repeat_interleave(repeat_factor, dim=0)
    
    # Check dimensions and copy if match
    actual_output_dim = weight.shape[0]
    assert actual_output_dim == expected_output_dim, \
        f"{proj_name} dimension mismatch: output_dim={actual_output_dim} != expected={expected_output_dim}"
    student_proj.weight.data.copy_(weight)
    stats["loaded_keys"].append(f"{proj_name}.weight")
    
    if bias is not None and student_proj.bias is not None:
        student_proj.bias.data.copy_(bias)
        stats["loaded_keys"].append(f"{proj_name}.bias")
    
    stats["transferred"].append(f"{proj_name} (from {proj_name[0].upper()})")
    return stats
