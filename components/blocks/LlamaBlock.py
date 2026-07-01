import torch.nn as nn
from torch import Tensor
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaMLP, LlamaRMSNorm

from components._factory import apply_module_factory_kwargs
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
        
        # Mixer
        MixerClass = Registry(config.Layer.name)
        self.mixer = MixerClass(
            d_model=self.d_model,
            layer_idx=layer_idx,
            **kwargs,
            **config.Layer.input,
            **factory_kwargs
        )

        # Other components
        norm_epsilon = self.config.input.norm_epsilon
        self.input_layernorm = LlamaRMSNorm(hidden_size=self.d_model, eps=norm_epsilon)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size=self.d_model, eps=norm_epsilon)
        self.mlp = LlamaMLP(
            LlamaConfig(
                hidden_size=self.d_model,
                intermediate_size=config.input.mlp_intermediate_size,
                hidden_act=config.input.mlp_act_fn,
            )
        )
        self.input_layernorm = apply_module_factory_kwargs(
            self.input_layernorm, factory_kwargs
        )
        self.post_attention_layernorm = apply_module_factory_kwargs(
            self.post_attention_layernorm, factory_kwargs
        )
        self.mlp = apply_module_factory_kwargs(self.mlp, factory_kwargs)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids=None,
        position_embeddings=None,
        inference_params=None,
        run_mlp_component=True,
        return_mixer_matrix=False,
        return_mixer_hidden_states=False,
        **kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """

        outputs = {}

        # Apply Mixer
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        mixer_outputs = self.mixer(
            hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            return_mixer_matrix=return_mixer_matrix,
            inference_params=inference_params,
        )

        mixer_outputs["hidden_states"] = mixer_outputs["hidden_states"].to(
            residual.dtype
        )

        if not run_mlp_component:
            return mixer_outputs

        hidden_states = mixer_outputs["hidden_states"] + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # store outputs
        if return_mixer_hidden_states:
            outputs["mixer_hidden_states"] = mixer_outputs["hidden_states"]
        if return_mixer_matrix:
            outputs["transfer_matrix"] = mixer_outputs["transfer_matrix"]
        outputs["hidden_states"] = hidden_states

        return outputs

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )
