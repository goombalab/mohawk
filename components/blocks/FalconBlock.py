# Copyright (c) 2023, Tri Dao, Albert Gu.

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from components._factory import apply_module_factory_kwargs
from components.registry import Registry
from transformers.models.falcon_mamba.modeling_falcon_mamba import (
    FalconMambaRMSNorm,
    FalconMambaConfig,
)

class Block(nn.Module):
    def __init__(
        self,
        d_model, 
        config, 
        factory_kwargs, 
        layer_idx=None,
        residual_in_fp32=True,
        **kwargs
    ):
        """
        This is a wrapper class for the FalconMambaMixer.
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.d_model = d_model
        self.config = config
        self.layer_idx = layer_idx
        self.input_layernorm = FalconMambaRMSNorm(hidden_size=self.d_model, eps=1e-5)
        self.input_layernorm = apply_module_factory_kwargs(
            self.input_layernorm, factory_kwargs
        )


        falcon_mamba_config = FalconMambaConfig(
            hidden_size=self.d_model,
        )

        self.mixer = Registry(config.Layer.name)(
            config=falcon_mamba_config,
            layer_idx=layer_idx,
        )
        self.mixer = apply_module_factory_kwargs(self.mixer, factory_kwargs)
        

    def forward(
        self,
        hidden_states: Tensor,
        inference_params=None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        output_dtype = hidden_states.dtype
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(
            hidden_states, 
            # inference_params=inference_params,
            # cache_position=cache_position, 
            # attention_mask=attention_mask
            )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states.to(output_dtype)
        return {"hidden_states": hidden_states}

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )
