from typing import Dict, List

import torch

from training_wrapper import BaseTrainingWrapper
from utils.config import Config
from utils.distributed import get_device


def _get_hidden_states(outputs):
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        hidden_states = getattr(outputs, "all_hidden_states", None)
    return hidden_states


def _get_attentions(outputs):
    attentions = getattr(outputs, "attentions", None)
    if attentions is None:
        attentions = getattr(outputs, "all_transfer_matrices", None)
    return attentions


def distill_step(
    batch: torch.Tensor,
    student_wrapper: BaseTrainingWrapper,
    teacher_wrapper: BaseTrainingWrapper,
    cfg: Config,
    **kwargs,
) -> List[float]:
    device = get_device()
    input_ids = batch["input_ids"].to(device)
    scores = []

    with torch.no_grad():
        teacher_outputs = teacher_wrapper.model(
            input_ids=input_ids,
            output_hidden_states=True,  # needed for input to student
            # output_attention_results=False,
            # output_logits=False,
            output_attentions=True,
            use_cache=False,
        )
    teacher_hidden_states = _get_hidden_states(teacher_outputs)
    teacher_attentions = _get_attentions(teacher_outputs)
    assert teacher_hidden_states is not None, "Teacher model did not return hidden states."
    assert teacher_attentions is not None, "Teacher model did not return attention matrices."
    num_layers = len(student_wrapper.module.backbone.layers)
    assert len(teacher_hidden_states) >= num_layers + 1, \
        f"Teacher model should have at least {num_layers + 1} hidden states (input plus layer outputs). Found {len(teacher_hidden_states)}"
    assert len(teacher_attentions) >= num_layers, \
        f"Teacher model should have at least {num_layers} attention matrices. Found {len(teacher_attentions)}"

    # Sum per-layer losses and call backward() once. Each layer's loss only
    # depends on that layer's parameters, so the per-parameter gradient is
    # identical to per-layer backward, but DDP's reducer requires a single
    # backward per iteration.
    total_loss = None
    for layer_idx in range(num_layers):
        input = teacher_hidden_states[layer_idx].to(device)

        # Forward pass
        student_wrapper.module.forward_fn = "_layer_forward"
        student_outputs = student_wrapper.model(
            layer_idx=layer_idx,
            hidden_states=input,
            run_mlp_component=False,
            return_mixer_matrix=True,
            return_hidden_states=False,
        )
        student_wrapper.module.forward_fn = "_forward"
        transfer_matrix = student_outputs["transfer_matrix"].to(device)
        attn_matrix = teacher_attentions[layer_idx].to(device)

        # Calculate mixer distance:
        assert transfer_matrix.size() == attn_matrix.size()
        matrix_distance = torch.linalg.matrix_norm(
            transfer_matrix - attn_matrix, ord="fro"
        ).mean()
        scores.append(matrix_distance.item())
        total_loss = matrix_distance if total_loss is None else total_loss + matrix_distance

    if total_loss is not None:
        student_wrapper.backward(total_loss)

    # Free memory
    teacher_outputs = student_outputs = None

    return scores
