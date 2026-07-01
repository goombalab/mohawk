import torch
import torch.nn as nn

from external_models.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, LlamaConfig
from components.registry import Registry
from components._repo_path import ensure_repo_root_on_path

ensure_repo_root_on_path()

from utils.config import Config


def _to_factory_dtype_device(module, factory_kwargs):
    move_kwargs = {k: v for k, v in factory_kwargs.items() if v is not None}
    if not move_kwargs:
        return module
    tensors = list(module.parameters(recurse=True)) + list(module.buffers(recurse=True))
    if any(tensor.device.type == "meta" for tensor in tensors):
        dtype = move_kwargs.get("dtype")
        return module.to(dtype=dtype) if dtype is not None else module
    return module.to(**move_kwargs)


class MixerModel(nn.Module):
    def __init__(
        self: nn.Module,
        input_size: int,
        config: Config,
        device=None, 
        dtype=None,
        **kwargs
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.config = config
        n_layer = self.config.input.n_layer
        d_model = self.config.input.d_model

        self.embedding = nn.Embedding(input_size, d_model, **factory_kwargs)
        self.rotary_emb = LlamaRotaryEmbedding(
            config=LlamaConfig(
                hidden_size=d_model,
                **self.config.input,
            ),
            device=device,
        )
        self.layers = nn.ModuleList()
        for block_cfg in self.config.Blocks:
            Block = Registry(block_cfg.name)
            n_layers = block_cfg.n_layers
            layers = nn.ModuleList(
                [
                    Block(
                        d_model=d_model,
                        config=block_cfg,
                        factory_kwargs=factory_kwargs,
                        layer_idx=i,
                        **kwargs,
                    )
                    for i in range(len(self.layers), len(self.layers) + n_layers)
                ]
            )
            self.layers += layers
        assert len(self.layers) == n_layer

        # Final layer norm
        norm_epsilon = self.config.input.norm_epsilon
        self.final_layernorm = LlamaRMSNorm(hidden_size=d_model, eps=norm_epsilon)
        self.final_layernorm = _to_factory_dtype_device(
            self.final_layernorm, factory_kwargs
        )

        return

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None, # in case the model uses position embeddings
        return_mixer_matrix=False,
        return_mixer_hidden_states=False,
        return_hidden_states=False,
        inference_params=None,
        **kwargs,
    ):
        assert (input_ids is None) != (inputs_embeds is None), "Only one of input_ids or inputs_embeds should be provided"
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        hidden_states = inputs_embeds

        # https://github.com/pytorch/pytorch/issues/98814:
        # nn.Embedding acts like a lookup table and won't apply autocast
        # if torch.is_autocast_enabled():
        #     auto_dtype = torch.get_autocast_gpu_dtype()
        #     hidden_states = hidden_states.to(auto_dtype)
        
        if position_ids is None:
            batch_size = inputs_embeds.shape[0]
            seq_len = inputs_embeds.shape[1]
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)

        # Prepare the position embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Initialize outputWs
        outputs = {
            "last_hidden_state": None,
            "position_embeddings": position_embeddings,
            "all_hidden_states": tuple(),
            "all_transfer_matrices": tuple(),
            "all_mixer_outputs": tuple(),
        }

        if return_hidden_states:
            outputs["all_hidden_states"] += (hidden_states,)

        # Run the layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                return_mixer_matrix=return_mixer_matrix,
                return_mixer_hidden_states=return_mixer_hidden_states,
                inference_params=inference_params,
                **kwargs,
            )

            # Record outputs
            hidden_states = layer_outputs["hidden_states"]
            if return_hidden_states:
                outputs["all_hidden_states"] += (hidden_states,)
            if return_mixer_hidden_states:
                outputs["all_mixer_outputs"] += (layer_outputs["mixer_hidden_states"],)
            if return_mixer_matrix and "transfer_matrix" in layer_outputs:
                outputs["all_transfer_matrices"] += (layer_outputs["transfer_matrix"],)

        # Last layer, apply layer norm
        outputs["last_hidden_state"] = self.final_layernorm(hidden_states)
        return outputs

    def _layer_forward(self, layer_idx, *args, **kwargs):
        return self.layers[layer_idx](*args, **kwargs)
