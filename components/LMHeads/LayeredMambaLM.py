# Copyright (c) 2023, Albert Gu, Tri Dao.

import json
import os
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.utils import ModelOutput
from huggingface_hub import PyTorchModelHubMixin
# from mamba_ssm.utils.generation import GenerationMixin
from generation.generation_mixin import GenerationMixin

from components.registry import Registry
from utils.config import Config


@dataclass
class CustomMambaCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    all_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_transfer_matrices: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_mixer_outputs: Optional[Tuple[torch.FloatTensor, ...]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    position_embeddings: Optional[torch.FloatTensor] = None


class LMHeadModel(
    nn.Module,
    GenerationMixin,
    PyTorchModelHubMixin,
):
    def __init__(
        self,
        config: Optional[Union[Dict, Config]],
        initializer_cfg=None,
        device=None,
        dtype=None,
        **kwargs
    ) -> None:

        # Load config (from_pretrained gives a dict)
        if not isinstance(config, Config):
            config = Config.from_dict(config)
        self.config = config

        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        # Pad vocab size to be a multiple of pad_vocab_size_multiple
        vocab_size = config.input.vocab_size
        pad_vocab_size_multiple = config.input.pad_vocab_size_multiple
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )
        self.config.input.vocab_size = vocab_size

        # Mixer model
        mixer_model_cfg = self.config.MixerModel
        MixerModel = Registry(mixer_model_cfg.name)
        self.backbone = MixerModel(
            input_size=vocab_size,
            config=mixer_model_cfg,
            initializer_cfg=initializer_cfg,
            **factory_kwargs,
            **kwargs
        )

        if not self.config.input.tie_embeddings:
            self.lm_head = nn.Linear(
                in_features=mixer_model_cfg.input.d_model,
                out_features=vocab_size,
                bias=self.config.input.lm_head_bias,
                **factory_kwargs,
            )
        else:
            self.lm_head = lambda x: x @ self.backbone.embedding.weight.t()

        # Forward function
        self.forward_fn = "_forward"

        return

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def set_forward_fn(self, forward_fn):
        self.forward_fn = forward_fn

    def forward(self, *args, **kwargs):
        return getattr(self, self.forward_fn)(*args, **kwargs)

    def _forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        return_mixer_matrix=False,
        return_mixer_hidden_states=False,
        return_hidden_states=False,
        return_logits=True,
        inference_params=None,
        num_last_tokens=0,
        **kwargs,
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        outputs = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            return_mixer_matrix=return_mixer_matrix,
            return_mixer_hidden_states=return_mixer_hidden_states,
            return_hidden_states=return_hidden_states,
            inference_params=inference_params,
            **kwargs,
        )

        if outputs["last_hidden_state"] is not None and return_logits:
            logits = self.lm_head(outputs["last_hidden_state"]) #.float()
            outputs["logits"] = (
                logits if num_last_tokens == 0 else logits[:, -num_last_tokens:]
            )
        else:
            outputs["logits"] = None

        return CustomMambaCausalLMOutput(
            loss=None,
            logits=outputs["logits"],
            all_hidden_states=outputs["all_hidden_states"],
            all_transfer_matrices=outputs["all_transfer_matrices"],
            all_mixer_outputs=outputs["all_mixer_outputs"],
            last_hidden_state=outputs["last_hidden_state"],
            position_embeddings=outputs["position_embeddings"],
        )

    def _layer_forward(self, *args, **kwargs):
        return self.backbone._layer_forward(*args, **kwargs)
