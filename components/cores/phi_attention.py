from torch import nn
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from external_models.modeling_phi import PHI_ATTENTION_CLASSES, PhiConfig


def _to_factory_dtype_device(module, factory_kwargs):
    move_kwargs = {k: v for k, v in factory_kwargs.items() if v is not None}
    if not move_kwargs:
        return module
    tensors = list(module.parameters(recurse=True)) + list(module.buffers(recurse=True))
    if any(tensor.device.type == "meta" for tensor in tensors):
        dtype = move_kwargs.get("dtype")
        return module.to(dtype=dtype) if dtype is not None else module
    return module.to(**move_kwargs)


class Mixer(nn.Module):
    def __init__(
            self: nn.Module,
            layer_idx: int,
            initializer: dict = None,
            **kwargs
            ):
        super().__init__()
        factory_kwargs = {
            "device": kwargs.pop("device", None),
            "dtype": kwargs.pop("dtype", None),
        }
        self.model_cfg = kwargs
        self.weight_init_cfg = initializer

        # Select the attention class based on the configuration
        type = "flash_attention_2" if self.model_cfg["flash_attention"] else "eager"
        self.self_attn = PHI_ATTENTION_CLASSES[type](
            PhiConfig(**self.model_cfg), layer_idx
        )
        self.self_attn = _to_factory_dtype_device(self.self_attn, factory_kwargs)
        self._attention_mask = None

    def forward(
        self, 
        hidden_states, 
        attention_mask=None, 
        return_mixer_matrix=False, 
        **kwargs
    ):

        if attention_mask is None:
            attention_mask = self.create_mask(hidden_states)

        result = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=return_mixer_matrix,
        )
        attn_output, attn_weights, past_key_value = result

        return {"hidden_states": attn_output, "transfer_matrix": attn_weights}

    def create_mask(self, hidden_states):
        batch_size, seq_length = hidden_states.size()[:2]
        # Is the attention mask already prepared?
        if (
            self._attention_mask is not None
            and self._attention_mask.size()[:2] == hidden_states.size()[:2]
        ):
            return self._attention_mask
        elif (
            self._attention_mask is not None
            and self._attention_mask.size()[:2] != hidden_states.size()[:2]
        ):
            self._attention_mask = None
        # Prepare the attention mask
        if self.model_cfg["flash_attention"]:
            # 2d mask is passed through the layers
            self._attention_mask = (
                self._attention_mask
                if (self._attention_mask is not None and 0 in self._attention_mask)
                else None
            )
        else:
            # 4d mask is passed through the layers
            self._attention_mask = _prepare_4d_causal_attention_mask(
                self._attention_mask, (batch_size, seq_length), hidden_states, 0
            )
        return self._attention_mask
