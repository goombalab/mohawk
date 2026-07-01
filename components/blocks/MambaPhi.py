import torch.nn as nn
from torch import Tensor
from transformers.models.phi.configuration_phi import PhiConfig

from components.registry import Registry
from external_models.modeling_phi import PhiMLP


def _to_factory_dtype_device(module, factory_kwargs):
    move_kwargs = {k: v for k, v in factory_kwargs.items() if v is not None}
    if not move_kwargs:
        return module
    tensors = list(module.parameters(recurse=True)) + list(module.buffers(recurse=True))
    if any(tensor.device.type == "meta" for tensor in tensors):
        dtype = move_kwargs.get("dtype")
        return module.to(dtype=dtype) if dtype is not None else module
    return module.to(**move_kwargs)


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
        self.input_layernorm = nn.LayerNorm(self.d_model, eps=1e-5, **factory_kwargs)
        self.mlp = PhiMLP(
            PhiConfig(
                hidden_size=self.d_model,
                intermediate_size=self.d_model * 4,
                hidden_act="gelu_new",
            )
        )
        self.mlp = _to_factory_dtype_device(self.mlp, factory_kwargs)
        self.resid_dropout = nn.Dropout(config.input.resid_dropout)

        return

    def forward(
        self,
        hidden_states: Tensor,
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

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Apply Mixer
        mixer_outputs = self.mixer(
            hidden_states,
            return_mixer_matrix=return_mixer_matrix,
            inference_params=inference_params,
        )
        mixer_outputs["hidden_states"] = mixer_outputs["hidden_states"].to(
            residual.dtype
        )

        if not run_mlp_component:
            return mixer_outputs

        # store outputs
        if return_mixer_hidden_states:
            outputs["mixer_hidden_states"] = mixer_outputs["hidden_states"]
        if return_mixer_matrix:
            outputs["transfer_matrix"] = mixer_outputs["transfer_matrix"]

        # Feed Forward
        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states)).to(
            residual.dtype
        )

        # Mixer output
        mixer_output = self.resid_dropout(mixer_outputs["hidden_states"])

        # sum all up (this is not sequential)
        outputs["hidden_states"] = mixer_output + feed_forward_hidden_states + residual

        return outputs

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        if getattr(self.mixer, "allocate_inference_cache", None) is None:
            return
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )
