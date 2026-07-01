from typing import Dict, List

import torch
import torch.nn.functional as F

from training_wrapper import BaseTrainingWrapper
from utils.config import Config
from utils.distributed import get_device


def get_logprobs(
    model_wrapper: BaseTrainingWrapper,
    input_ids: torch.Tensor,
    response_ids: torch.Tensor,
    position_ids: torch.Tensor = None,
    vocab_size: int = None,
) -> torch.Tensor:
    """
    Compute log probabilities of response_ids given input_ids.
    
    Args:
        model_wrapper: Model to compute logprobs from
        input_ids: Prompt tokens [batch_size, prompt_len]
        response_ids: Response tokens [batch_size, response_len]
        position_ids: Optional position IDs
        vocab_size: Vocabulary size
        
    Returns:
        Log probabilities per sample [batch_size]
    """
    # Concatenate prompt and response
    total_ids = torch.cat([input_ids, response_ids], dim=1)
    
    # Get logits from model
    if hasattr(model_wrapper.module, 'forward_fn'):
        model_wrapper.module.forward_fn = "_forward"
    
    outputs = model_wrapper(
        input_ids=total_ids,
        position_ids=position_ids,
        return_hidden_states=False,
        return_mixer_matrix=False,
        use_cache=False,
    )
    
    logits = outputs.logits  # [batch_size, seq_len, vocab_size]
    
    # Extract logits for response tokens only
    # Shift by 1 for next-token prediction: logits[i] predicts token[i+1]
    prompt_len = input_ids.size(1)
    response_len = response_ids.size(1)
    
    # Get logits for response tokens: logits[prompt_len-1] predicts response[0]
    response_logits = logits[:, prompt_len-1:prompt_len-1+response_len, :]  # [batch_size, response_len, vocab_size]
    
    # Compute log probabilities
    log_probs = F.log_softmax(response_logits, dim=-1)  # [batch_size, response_len, vocab_size]
    
    # Gather log probabilities for actual response tokens
    response_ids_flat = response_ids.view(-1)  # [batch_size * response_len]
    log_probs_flat = log_probs.reshape(-1, vocab_size)  # [batch_size * response_len, vocab_size]
    
    # Get log prob for each token in response
    token_logprobs = log_probs_flat.gather(1, response_ids_flat.unsqueeze(1)).squeeze(1)
    token_logprobs = token_logprobs.view(response_ids.shape)  # [batch_size, response_len]
    
    # Mask out padding tokens (assuming pad_token_id is 0 or a specific value)
    # For now, we'll sum all tokens - in practice, you may want to add explicit masking
    # if your tokenizer uses a specific pad_token_id
    
    # Sum log probabilities (product in log space) per sequence
    logprobs = token_logprobs.sum(dim=1)  # [batch_size]
    
    return logprobs


def distill_step(
    batch: Dict[str, torch.Tensor],
    student_wrapper: BaseTrainingWrapper,
    teacher_wrapper: BaseTrainingWrapper,
    cfg: Config,
    beta: float = 0.1,
    # Flexible key mapping for different dataset formats
    prompt_key: str = None,  # Auto-detect if None
    chosen_key: str = None,
    rejected_key: str = None,
    **kwargs,
) -> List[float]:
    """
    DPO (Direct Preference Optimization) distillation step.
    
    Loss: -log(sigma(beta * (log_p_student_chosen - log_p_student_rejected 
                             - log_p_ref_chosen + log_p_ref_rejected)))
    
    Args:
        batch: Dictionary containing:
            - input_ids: Prompt tokens [batch_size, prompt_len] (or prompt_key)
            - chosen_ids: Preferred response tokens [batch_size, chosen_len] (or chosen_key)
            - rejected_ids: Rejected response tokens [batch_size, rejected_len] (or rejected_key)
            - position_ids: Optional position IDs
        student_wrapper: Student model (being optimized)
        teacher_wrapper: Reference model (teacher, frozen)
        cfg: Configuration object
        beta: DPO beta parameter (temperature for the sigmoid)
        prompt_key: Key name for prompt in batch (auto-detects if None)
        chosen_key: Key name for chosen response in batch (auto-detects if None)
        rejected_key: Key name for rejected response in batch (auto-detects if None)
        **kwargs: Additional arguments (e.g., tokenizer)
        
    Returns:
        List of loss values
    """
    # Try to find the keys in the batch (similar to collate_fns.py pattern)
    # Prompt keys
    if prompt_key is None:
        prompt_keys = ["input_ids", "prompt", "query", "context"]
        for key in prompt_keys:
            if key in batch:
                prompt_key = key
                break
    if prompt_key not in batch:
        raise ValueError(f"Could not find prompt in batch. Available keys: {list(batch.keys())}")
    device = get_device()
    input_ids = batch[prompt_key].to(device)
    
    # Chosen keys
    if chosen_key is None:
        chosen_keys = ["chosen_ids", "chosen", "preferred", "response"]
        for key in chosen_keys:
            if key in batch:
                chosen_key = key
                break
    if chosen_key not in batch:
        raise ValueError(f"Could not find chosen response in batch. Available keys: {list(batch.keys())}")
    chosen_ids = batch[chosen_key].to(device)
    
    # Rejected keys
    if rejected_key is None:
        rejected_keys = ["rejected_ids", "rejected", "dispreferred"]
        for key in rejected_keys:
            if key in batch:
                rejected_key = key
                break
    if rejected_key not in batch:
        raise ValueError(f"Could not find rejected response in batch. Available keys: {list(batch.keys())}")
    rejected_ids = batch[rejected_key].to(device)
    
    position_ids = batch.get("position_ids", None)
    if position_ids is not None:
        position_ids = position_ids.to(device)
    
    vocab_size = cfg.ComponentsConfig.input.vocab_size
    
    # Get log probabilities from student model
    student_chosen_logp = get_logprobs(
        student_wrapper, input_ids, chosen_ids, position_ids, vocab_size
    )
    student_rejected_logp = get_logprobs(
        student_wrapper, input_ids, rejected_ids, position_ids, vocab_size
    )
    
    # Get log probabilities from reference model (teacher, frozen)
    with torch.no_grad():
        ref_chosen_logp = get_logprobs(
            teacher_wrapper, input_ids, chosen_ids, position_ids, vocab_size
        )
        ref_rejected_logp = get_logprobs(
            teacher_wrapper, input_ids, rejected_ids, position_ids, vocab_size
        )
    
    # Compute DPO loss
    # log(π_student(y_w|x) / π_student(y_l|x)) - log(π_ref(y_w|x) / π_ref(y_l|x))
    log_ratio = (student_chosen_logp - student_rejected_logp) - \
                (ref_chosen_logp - ref_rejected_logp)
    
    # DPO loss: -log(sigma(beta * log_ratio))
    # This encourages log_ratio > 0 (student prefers chosen over rejected)
    loss = -F.logsigmoid(beta * log_ratio).mean()
    
    # Backward pass
    student_wrapper.backward(loss)
    
    return [loss.item()]
