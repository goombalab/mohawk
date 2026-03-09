from typing import Dict, List

import torch

from training_wrapper import BaseTrainingWrapper
from utils.config import Config
from utils.distributed import local_rank


def distill_step(
    batch: Dict[str, torch.Tensor],
    student_wrapper: BaseTrainingWrapper,
    teacher_wrapper: BaseTrainingWrapper,
    cfg: Config,
    hstates_gap: int = 1,
    **kwargs,
) -> List[float]:
    input_ids = batch["input_ids"].to(local_rank)
    
    num_layers = len(student_wrapper.module.backbone.layers)
    assert num_layers % hstates_gap == 0

    scores = []

    teacher_outputs = teacher_wrapper(
        input_ids=input_ids,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
    )

    layer_idx = 0
    while layer_idx < num_layers:
        
        # Get student hidden states
        hidden_states = teacher_outputs.hidden_states[layer_idx]

        # Forward pass
        student_wrapper.module.forward_fn = "_layer_forward"
        for _ in range(hstates_gap):
            hidden_states = student_wrapper(
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                return_mixer_matrix=False,
                return_hidden_states=False,
            )["hidden_states"]
            layer_idx += 1
        student_wrapper.module.forward_fn = "_forward"

        # Get teacher hidden states
        output = teacher_outputs.hidden_states[layer_idx]
        assert hidden_states.size() == output.size()

        # Calculate hstates distance:
        hstates_distance = torch.norm(hidden_states - output, p=2, dim=(-1,)).mean()
        scores.append(hstates_distance.item())

        # Free memory
        # hidden_states = input = output = None

        # Backward pass
        student_wrapper.backward(hstates_distance)

    return scores
