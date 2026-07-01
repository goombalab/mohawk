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


def distill_step(
    batch: Dict[str, torch.Tensor],
    student_wrapper: BaseTrainingWrapper,
    teacher_wrapper: BaseTrainingWrapper,
    cfg: Config,
    hstates_gap: int = 1,
    **kwargs,
) -> List[float]:
    device = get_device()
    input_ids = batch["input_ids"].to(device)
    if "position_ids" in batch:
        position_ids = batch["position_ids"].to(device)
    else:
        position_ids = None
    
    num_layers = len(student_wrapper.module.backbone.layers)
    assert num_layers % hstates_gap == 0

    scores = []

    with torch.no_grad():
        teacher_outputs = teacher_wrapper(
            input_ids=input_ids,
            output_hidden_states=True,
            position_ids=position_ids,
            output_attentions=False,
            use_cache=False,
        )
    teacher_hidden_states = _get_hidden_states(teacher_outputs)
    assert teacher_hidden_states is not None, "Teacher model did not return hidden states."
    assert len(teacher_hidden_states) >= num_layers + 1, \
        f"Teacher model should have at least {num_layers + 1} hidden states (input plus layer outputs). Found {len(teacher_hidden_states)}"

    # HF GPT-2's CausalLMOutputWithCrossAttentions doesn't expose
    # position_embeddings; the student's _layer_forward accepts None and will
    # compute its own RoPE from position_ids.
    teacher_pos_emb = getattr(teacher_outputs, "position_embeddings", None)

    total_loss = None
    layer_idx = 0
    while layer_idx < num_layers:

        # Get student hidden states
        hidden_states = teacher_hidden_states[layer_idx]

        # Forward pass
        student_wrapper.module.forward_fn = "_layer_forward"
        for _ in range(hstates_gap):
            hidden_states = student_wrapper(
                layer_idx=layer_idx,
                position_ids=position_ids,
                position_embeddings=teacher_pos_emb,
                hidden_states=hidden_states,
                return_mixer_matrix=False,
                return_hidden_states=False,
            )["hidden_states"]
            layer_idx += 1
        student_wrapper.module.forward_fn = "_forward"

        # Get teacher hidden states
        output = teacher_hidden_states[layer_idx]
        assert hidden_states.size() == output.size()

        # Calculate hstates distance:
        hstates_distance = torch.norm(hidden_states - output, p=2, dim=(-1,), dtype=torch.float32).mean()
        scores.append(hstates_distance.item())


        total_loss = hstates_distance if total_loss is None else total_loss + hstates_distance

    if total_loss is not None:
        student_wrapper.backward(total_loss)

    return scores
