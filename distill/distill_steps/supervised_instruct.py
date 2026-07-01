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
    generation_max_new_tokens: int = 32,
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
    device = get_device()
    context = batch["input_ids"].to(device)
    response = batch["response_ids"].to(device)
    vocab_size = cfg.ComponentsConfig.input.vocab_size
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Teacher forward pass
    if supervision and generate:
        # response is a vector of ids sampled from the teacher.
        generated_responses = []
        max_positions = cfg.ComponentsConfig.MixerModel.input.get(
            "max_position_embeddings",
            context.size(1) + generation_max_new_tokens,
        )
        for sample in context:
            sample = sample[sample != pad_token_id].unsqueeze(0)
            max_length = min(sample.size(1) + generation_max_new_tokens, max_positions)
            if max_length <= sample.size(1):
                raise ValueError(
                    "generation_max_new_tokens leaves no room for generated response tokens "
                    f"with prompt length {sample.size(1)} and max_position_embeddings {max_positions}."
                )
            teacher_preds = teacher_wrapper.generate(
                sample,
                max_length=max_length,
                use_cache=False,
            )
            generated_responses.append(teacher_preds[0, sample.size(1):])
        response = torch.nn.utils.rnn.pad_sequence(
            generated_responses,
            batch_first=True,
            padding_value=pad_token_id,
        ).to(device)
        total_ids = torch.cat([context, response], dim=1)
    elif supervision and not generate:
        # response supervision comes from the teacher distribution over the response span.
        total_ids = torch.cat([context, response], dim=1)
        teacher_logits = teacher_wrapper(input_ids=total_ids, use_cache=False).logits
        prompt_len = context.size(1)
        response_len = response.size(1)
        response = teacher_logits[:, prompt_len - 1:prompt_len - 1 + response_len]
    else:
        # keep response as it is
        total_ids = torch.cat([context, response], dim=1)

    # Student forward pass
    student_logits = student_wrapper(total_ids).logits
    prompt_len = context.size(1)
    response_len = response.size(1)
    student_logits = student_logits[:, prompt_len - 1:prompt_len - 1 + response_len].reshape(-1, vocab_size)

    # Cross-entropy only on the response part:
    if supervision and not generate:
        supervised_ce = cross_entropy(student_logits, response.reshape(-1, vocab_size), cfg.DistillConfig.temperature)
    else:
        supervised_ce = torch.nn.functional.cross_entropy(student_logits.float(), response.reshape(-1))

    # Backward pass
    student_wrapper.backward(supervised_ce)

    return [torch.exp(supervised_ce).item()]
