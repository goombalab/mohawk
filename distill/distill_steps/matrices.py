from typing import Dict, List

import torch

from training_wrapper import BaseTrainingWrapper
from utils.config import Config
from utils.distributed import local_rank, world_size


def distill_step(
    batch: torch.Tensor,
    student_wrapper: BaseTrainingWrapper,
    teacher_wrapper: BaseTrainingWrapper,
    cfg: Config,
    **kwargs,
) -> List[float]:
    input_ids = batch["input_ids"].to(local_rank)
    scores = []

    teacher_outputs = teacher_wrapper.model(
        input_ids=input_ids,
        output_hidden_states=True,  # needed for input to student
        # output_attention_results=False,
        # output_logits=False,
        output_attentions=True,
        use_cache=False,
    )

    for layer_idx in range(len(student_wrapper.module.backbone.layers)):
        input = teacher_outputs.hidden_states[layer_idx].to(local_rank)

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
        transfer_matrix = student_outputs["transfer_matrix"].to(local_rank)
        attn_matrix = teacher_outputs.attentions[layer_idx].to(local_rank)

        # Calculate mixer distance:
        assert transfer_matrix.size() == attn_matrix.size()
        matrix_distance = torch.linalg.matrix_norm(
            transfer_matrix - attn_matrix, ord="fro"
        ).mean()
        scores.append(matrix_distance.item())

        # Backward pass
        student_wrapper.backward(matrix_distance)

    # Free memory
    teacher_outputs = student_outputs = None

    return scores
