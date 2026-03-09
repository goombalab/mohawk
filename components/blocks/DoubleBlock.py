import torch.nn as nn
from torch import Tensor
MLP, MLPConfig, RMSNorm = None, None, None
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaMLP,
    LlamaRMSNorm,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config, 
    Qwen2MLP, 
    Qwen2RMSNorm,
)
import torch
from components.registry import Registry

class Block(nn.Module):
    def __init__(self, d_model, config, factory_kwargs, layer_idx, **kwargs):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.d_model = d_model
        self.config = config
        self.layer_idx = layer_idx

        self.head_dim = config.ssm_layer.input.headdim
        self.ssm_heads = config.ssm_layer.input.n_v_heads
        self.att_heads = config.attn_layer.input.num_attention_heads if config.get("attn_layer", None) is not None else 0
        

        # SSM
        MixerClass = Registry(config.ssm_layer.name)
        self.mixer1 = MixerClass(
            d_model=self.d_model,
            d_inner=self.head_dim*self.ssm_heads,
            layer_idx=layer_idx,
            **kwargs,
            **config.ssm_layer.input,
            **factory_kwargs
        )
        
        # Attention
        attn_layer_name = config.attn_layer.name
        if self.att_heads > 0:
            AttnClass = Registry(attn_layer_name)
            self.mixer2 = AttnClass(
                d_model=self.d_model,
                layer_idx=layer_idx,
                **kwargs,
                **config.attn_layer.input,
                **factory_kwargs
            )
        
        if attn_layer_name == "LlamaAttention":
            RMSNorm = LlamaRMSNorm
            MLP = LlamaMLP
            MLPConfig = LlamaConfig
        elif attn_layer_name == "Qwen2Attention":
            RMSNorm = Qwen2RMSNorm
            MLP = Qwen2MLP
            MLPConfig = Qwen2Config
        else:
            raise ValueError(f"Attention layer {attn_layer_name} not supported")
        
        # Other components
        self.input_layernorm = RMSNorm(hidden_size=self.d_model, eps=1e-5)
        self.post_attention_layernorm = RMSNorm(hidden_size=self.d_model, eps=1e-5)
        self.mlp = MLP(
            MLPConfig(
                hidden_size=self.d_model,
                intermediate_size=config.input.mlp_intermediate_size,
                hidden_act=config.input.mlp_act_fn,
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        ssm_cache = self.mixer1.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )
        if hasattr(self, "mixer2"):
            attn_cache = self.mixer2.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            return {"mixer1": ssm_cache, "mixer2": attn_cache}

        return {"mixer1": ssm_cache}
        
    def apply_mixers(self, *args, **kwargs):
        raise NotImplementedError("apply_mixers not implemented in base class")

    # def forward(self, hidden_states: Tensor, **kwargs):
    def forward(
        self,
        hidden_states: Tensor,
        position_ids=None,
        position_embeddings=None,
        inference_params=None,
        **kwargs
    ):
        hidden_states = hidden_states + self.apply_mixers(
            hidden_states, 
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            inference_params=inference_params,
            **kwargs
            ).to(hidden_states.dtype)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return {"hidden_states": hidden_states}
    

############################################################################
############################################################################
############################################################################
class Normalize(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    
    def __repr__(self):
        return f"normalize(eps={self.eps})"

    def forward(self, out_attn, out_mamba):
        """
        Scales the attention output to match the Mamba output.
        """
        mean_attn = out_attn.mean(dim=-1, keepdim=True)  # mean over hidden dim
        std_attn = out_attn.std(dim=-1, keepdim=True)    # std over hidden dim

        mean_mamba = out_mamba.mean(dim=-1, keepdim=True)
        std_mamba = out_mamba.std(dim=-1, keepdim=True)

        # Stabilized normalization
        out_attn_unit = (out_attn - mean_attn) / (std_attn + self.eps)  # zero mean, unit variance (per token)
        out_attn_scaled = out_attn_unit * (std_mamba + self.eps) + mean_mamba  # match mamba's per-token stats
        return out_attn_scaled

class Adapter(nn.Module):
    def __init__(self, d_model, config):
        super().__init__()
        self.d_model = d_model
        self.config = config
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.norm = Normalize(eps=1e-5)

    def forward(self, att_hidden_state, ssm_hidden_state, hidden_states):
        att_hidden_state = self.norm(att_hidden_state, ssm_hidden_state)
        att_hidden_state = self.out_proj(att_hidden_state)
        return att_hidden_state
    
class DoubleBlockAdapter(Block):
    def __init__(self, d_model, config, factory_kwargs, layer_idx, **kwargs):
        super().__init__(d_model, config, factory_kwargs, layer_idx, **kwargs)
        if hasattr(self, "mixer2"):
            self.adapter = Adapter(d_model, config)
        
    def apply_mixers(self, hidden_states, position_ids=None, position_embeddings=None, inference_params=None, **kwargs):
        hidden_states = self.input_layernorm(hidden_states)

        # Apply Mixer1
        ssm_outputs = self.mixer1(
            hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            inference_params=inference_params,
            **kwargs,
        )
        if isinstance(ssm_outputs, Tensor):
            ssm_outputs = {"hidden_states": ssm_outputs}
        ssm_hidden_state = ssm_outputs["hidden_states"]
        
        # Apply Mixer2 (if present)
        if hasattr(self, "mixer2"):
            att_output = self.mixer2(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                inference_params=inference_params,
                **kwargs,
            )
            assert ssm_outputs["hidden_states"].shape == att_output["hidden_states"].shape
            att_hidden_state = att_output["hidden_states"]
            att_hidden_state = self.adapter(att_hidden_state, ssm_hidden_state, hidden_states)
        else:
            att_hidden_state = None


        # Residual connection
        if att_hidden_state is not None:
            return (ssm_hidden_state + att_hidden_state) * 0.5
        
        return ssm_hidden_state

############################################################################
############################################################################
############################################################################
class DoubleBlockVanilla(Block):
    def __init__(self, d_model, config, factory_kwargs, layer_idx, **kwargs):
        super().__init__(d_model, config, factory_kwargs, layer_idx, **kwargs)

    def apply_mixers(self, hidden_states, position_ids=None, position_embeddings=None, **kwargs):
        hidden_states = self.input_layernorm(hidden_states)

        # Apply Mixer1
        ssm_outputs = self.mixer1(
            hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if isinstance(ssm_outputs, Tensor):
            ssm_outputs = {"hidden_states": ssm_outputs}

        if not hasattr(self, "mixer2"):
            return ssm_outputs["hidden_states"]
        
        att_output = self.mixer2(
            hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        assert ssm_outputs["hidden_states"].shape == att_output["hidden_states"].shape
        return ssm_outputs["hidden_states"] + att_output["hidden_states"]
    
############################################################################
############################################################################
############################################################################
class DoubleBlockHymba(Block):
    def __init__(self, d_model, config, factory_kwargs, layer_idx, **kwargs):
        super().__init__(d_model, config, factory_kwargs, layer_idx, **kwargs)
        RMSNorm = LlamaRMSNorm if config.attn_layer.name == "LlamaAttention" else Qwen2RMSNorm
        self.ssm_output_layernorm = RMSNorm(hidden_size=self.d_model, eps=1e-5)
        self.att_output_layernorm = RMSNorm(hidden_size=self.d_model, eps=1e-5)
        self.ssm_gate = nn.Parameter(torch.zeros(d_model))
        self.att_gate = nn.Parameter(torch.zeros(d_model))
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)

    def apply_mixers(self, hidden_states, position_ids=None, position_embeddings=None, **kwargs):
        hidden_states = self.input_layernorm(hidden_states)

        # Apply Mixer1
        ssm_outputs = self.mixer1(
            hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if isinstance(ssm_outputs, Tensor):
            ssm_outputs = {"hidden_states": ssm_outputs}
        ssm_hidden_state = self.ssm_output_layernorm(ssm_outputs["hidden_states"])
        ssm_hidden_state = self.ssm_gate * ssm_hidden_state

        if hasattr(self, "mixer2"):
            att_output = self.mixer2(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            att_hidden_state = self.att_output_layernorm(att_output["hidden_states"])
            att_hidden_state = self.att_gate * att_hidden_state

        else:
            att_hidden_state = None
        
        if att_hidden_state is not None:
            assert att_hidden_state.shape == ssm_hidden_state.shape
            return self.out_proj((ssm_hidden_state + att_hidden_state) * 0.5)
        
        return self.out_proj(ssm_hidden_state)

    
############################################################################
############################################################################
############################################################################
class DoubleRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.ssm_weight = nn.Parameter(torch.ones(hidden_size))
        self.attn_weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.ssm_weight * hidden_states.to(input_dtype), self.attn_weight * hidden_states.to(input_dtype)
    

class DoubleBlockMerger(Block):
    def __init__(self, d_model, config, factory_kwargs, layer_idx, **kwargs):
        super().__init__(d_model, config, factory_kwargs, layer_idx, **kwargs)
        RMSNorm = LlamaRMSNorm if config.attn_layer.name == "LlamaAttention" else Qwen2RMSNorm
        self.input_layernorm = DoubleRMSNorm(hidden_size=self.d_model, eps=1e-5)
        self.output_layernorm = RMSNorm(hidden_size=self.d_model, eps=1e-5)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)

    def apply_mixers(self, hidden_states, position_ids=None, position_embeddings=None, **kwargs):
        ssm_hidden_states, att_hidden_states = self.input_layernorm(hidden_states)

        # Apply Mixer1
        ssm_outputs = self.mixer1(
            ssm_hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if isinstance(ssm_outputs, Tensor):
            ssm_outputs = {"hidden_states": ssm_outputs}
        ssm_hidden_state = ssm_outputs["hidden_states"]
        ssm_hidden_state = self.out_proj(self.output_layernorm(ssm_hidden_state))
        
        # Apply Mixer2 (if present)
        if hasattr(self, "mixer2"):
            att_output = self.mixer2(
                att_hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            assert ssm_outputs["hidden_states"].shape == att_output["hidden_states"].shape
            att_hidden_state = att_output["hidden_states"]
            att_hidden_state = self.out_proj(self.output_layernorm(att_hidden_state))
        else:
            att_hidden_state = None


        # Residual connection
        if att_hidden_state is not None:
            return (ssm_hidden_state + att_hidden_state) * 0.5

        return ssm_hidden_state
    
############################################################################
############################################################################
############################################################################
