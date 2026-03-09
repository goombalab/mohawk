import torch
from torch import nn
# from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Config, Qwen2RotaryEmbedding, DynamicCache
from external_models.modeling_qwen2 import Qwen2Attention, Qwen2Config, Qwen2RotaryEmbedding, DynamicCache

class Mixer(nn.Module):
    def __init__(
            self: nn.Module,
            layer_idx: int,
            d_model: int,
            device: torch.device,
            dtype: torch.dtype,
            attn_implementation: str = "flash_attention_2",
            **kwargs
            ):
        super().__init__()
        self.model_cfg = kwargs

        # Select the attention class based on the configuration
        self.model_cfg["hidden_size"] = d_model
        if "_attn_implementation" not in self.model_cfg:
            self.model_cfg["_attn_implementation"] = attn_implementation
        config = Qwen2Config(**self.model_cfg)
        self.layer_idx = layer_idx
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx).to(dtype)
        self._attention_mask = None
        self.rotary_emb = Qwen2RotaryEmbedding(config=config, device=device)

    def forward(
        self,
        hidden_states,
        inference_params=None,
        attention_mask=None,
        return_mixer_matrix=False,
        position_ids=None,
        position_embeddings=None,
        **kwargs
    ):
        if inference_params is not None:
            # A slightly hacky way to handle the     cache (#layers instances of DynamicCache)
            if self.layer_idx not in inference_params.key_value_memory_dict:
                # Initialize the cache for the current layer
                inference_params.key_value_memory_dict[self.layer_idx] = {}
            if "past_key_value" not in inference_params.key_value_memory_dict[self.layer_idx]:
                # Initialize the past key value cache
                inference_params.key_value_memory_dict[self.layer_idx]['past_key_value'] = DynamicCache()
            # Retrieve the cache for the current layer
            past_key_value = inference_params.key_value_memory_dict[self.layer_idx]['past_key_value']
        else:
            past_key_value = None

        # Prepare the cache position
        past_seen_tokens = past_key_value.get_seq_length(self.layer_idx) if past_key_value is not None else 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + hidden_states.size(1), device=hidden_states.device)

        # Prepare the position ids
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Prepare the attention mask
        if attention_mask is None:
            attention_mask  = self._update_causal_mask(attention_mask, hidden_states, cache_position)

        if position_embeddings is None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Handle flash attention cu_seq_lens to avoid position_ids bug during generation decoding
        position_ids_for_attn = position_ids
        if self.model_cfg["_attn_implementation"] == "flash_attention_2" and inference_params is not None and inference_params.seqlen_offset > 0:
            position_ids_for_attn = None
            batch_size, seq_len = hidden_states.shape[:2]
            # Assert that all sequences are length 1 (as in generation decoding)
            assert seq_len == 1, f"cu_seq_lens computation assumes seq_len=1, but got {seq_len}"
            cu_seq_lens = torch.arange(0, batch_size + 1, dtype=torch.int32, device=hidden_states.device)
            kwargs['cu_seq_lens_q'] = cu_seq_lens
            kwargs['cu_seq_lens_k'] = cu_seq_lens

        # Perform the attention operation
        attn_output, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids_for_attn,
            past_key_value=past_key_value,
            output_attentions=return_mixer_matrix,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # assert past_key_value is None, "Past key value is not supported"
        return {"hidden_states": attn_output, "transfer_matrix": attn_weights}

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.model_cfg["_attn_implementation"] == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if hasattr(self.self_attn, "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[-1] + 1
            )

        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(
            target_length, device=device
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(
            input_tensor.shape[0], 1, -1, -1
        )

        return causal_mask

    def allocate_inference_cache(self, *args, **kwargs):
        return DynamicCache()
