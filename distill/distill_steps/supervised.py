from typing import Dict, List

import torch

from training_wrapper import BaseTrainingWrapper
from utils.config import Config
from utils.distributed import get_device


@torch.jit.script
def cross_entropy(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Cross-entropy loss.
    """
    return torch.nn.functional.cross_entropy(
        input=(student_logits / temperature).float(),
        target=teacher_logits.float().softmax(dim=-1),
    )

def distill_step(
    batch: Dict[str, torch.Tensor],
    student_wrapper: BaseTrainingWrapper,
    teacher_wrapper: BaseTrainingWrapper,
    cfg: Config,
    temperature: float = 1.0,
    **kwargs,
) -> List[float]:
    """
    Distillation step for supervised learning.

    Args:
        input_ids: Input tensor.
        student_wrapper: Student model.
        teacher_wrapper: Teacher model.
        cfg: Configuration object.
        temperature: scales the logits, reducing or increasing the model's confidence based on the temperature.
            If  T > 1 , the model becomes more uncertain (softer logits).
            If  T < 1 , the model becomes more confident (sharper logits).
    """
    device = get_device()
    input_ids = batch["input_ids"].to(device)
    vocab_size = cfg.ComponentsConfig.input.vocab_size
    if "position_ids" in batch:
        position_ids = batch["position_ids"].to(device)
    else:
        position_ids = None
        
    # Teacher forward pass
    with torch.no_grad():
        teacher_logits = teacher_wrapper(
            input_ids=input_ids,
            position_ids=position_ids,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
            ).logits.view(-1, vocab_size)

    # Student forward pass
    student_wrapper.module.forward_fn = "_forward"
    student_logits = student_wrapper(
        input_ids=input_ids,
        position_ids=position_ids,
        return_hidden_states=False,
        return_mixer_matrix=False,
        ).logits.view(-1, vocab_size)

    # Cross-entropy
    supervised_ce = cross_entropy(student_logits, teacher_logits, temperature)

    # Backward pass
    student_wrapper.backward(supervised_ce)

    return [torch.exp(supervised_ce).item()]
