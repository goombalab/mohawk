from typing import Dict, List

import torch

from training_wrapper import BaseTrainingWrapper
from utils.config import Config
from utils.distributed import local_rank


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
        input=student_logits / temperature,
        target=teacher_logits.softmax(dim=-1),
    )

def distill_step(
    batch: Dict[str, torch.Tensor],
    student_wrapper: BaseTrainingWrapper,
    teacher_wrapper: BaseTrainingWrapper,
    cfg: Config,
    tokenizer,
    generate: bool = False,
    supervision: bool = False,
    **kwargs,
) -> List[float]:
    """
    Distillation step for supervised instruction following.

    Args:
        input_ids: Input tensor.
        student_wrapper: Student model.
        teacher_wrapper: Teacher model.
        cfg: Configuration object.
        temperature: scales the logits, reducing or increasing the model's confidence based on the temperature.
            If  T > 1 , the model becomes more uncertain (softer logits).
            If  T < 1 , the model becomes more confident (sharper logits).
    """
    context = batch["input_ids"].to(local_rank)
    response = batch["response_ids"].to(local_rank)
    vocab_size = cfg.ComponentsConfig.input.vocab_size

    # Teacher forward pass
    if supervision and generate:
        # response is a vector of ids sampled from the teacher
        response = []
        for sample in context:
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            sample = sample[sample != pad_token_id].unsqueeze(0)
            teacher_preds = teacher_wrapper.generate(sample, max_length=sample.size(1)+32, use_cache=False)
            response.append(teacher_preds[0])
        response = torch.stack(response).to(local_rank)
    elif supervision and not generate:
        # response is a vector of logits
        teacher_logits = teacher_wrapper(total_ids, use_cache=False).logits
        response = teacher_logits[:, :, -response.size(1):].softmax(dim=-1)
    else:
        # keep response as it is
        pass
    
    total_ids = torch.cat([context, response], dim=1)

    # Student forward pass
    student_logits = student_wrapper(total_ids).logits
    student_logits = student_logits[:, -response.size(1):].reshape(-1, vocab_size)

    # Cross-entropy only on the response part:
    supervised_ce = torch.nn.functional.cross_entropy(student_logits, response.view(-1))

    # Backward pass
    student_wrapper.backward(supervised_ce)

    return [torch.exp(supervised_ce).item()]
