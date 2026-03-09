from functools import partial
import torch

# Define the common tokenizer arguments
common_tokenizer_args = {
    "padding": True,
    "truncation": True,
    "return_tensors": "pt",
    "return_attention_mask": False,
}

# Define the collate function
def setup_collate_fn(collate_type, tokenizer, chat_template=None, add_generation_prompt=False, return_dict=False):
    """
    Setup the collate function for the DataLoader.
    """
    # tokenizer.add_special_tokens({"additional_special_tokens": ["<|human|>", "<|gpt|>"]}) # hurts the embedding layer
    if collate_type == "conversation":
        return partial(collate_fn_conversation, tokenizer=tokenizer, chat_template=chat_template, add_generation_prompt=add_generation_prompt)
    elif collate_type == "instruction":
        return partial(collate_fn_instruction, tokenizer=tokenizer, return_dict=return_dict)
    elif collate_type == "text":
        return partial(collate_fn_text, tokenizer=tokenizer)
    elif collate_type == "raw":
        # return lambda x: torch.cat(x, dim=0)
        return lambda x: tokenizer(x, **common_tokenizer_args)["input_ids"]
    elif collate_type == "classic":
        return lambda x: torch.cat([torch.Tensor(item["input_ids"]) for item in x], dim=0).to(torch.int)
    elif collate_type == "preference":
        return partial(collate_fn_preference, tokenizer=tokenizer, chat_template=chat_template, add_generation_prompt=add_generation_prompt)
    else:
        raise ValueError(f"Dataset type {collate_type} not found.")

def collate_fn_conversation(batch, tokenizer, chat_template=None, add_generation_prompt=False):
    """
    Collate function for conversation datasets.
    A conversation template is for multi-turn tasks (back-and-forth dialogs, user/assistant roles, chat models).
    """

    role_dict = {
        "system": "system",
        "human": "user", 
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "bot": "assistant",
        "agent": "assistant",
        }
    
    keys = ["conversations", "messages", "text"]
    conversations_key = next((key for key in keys if key in batch[0].keys()), None)

    keys = ["from", "role"]
    role_key = next((key for key in keys if key in batch[0][conversations_key][0].keys()), None)

    keys = ["content", "value"]
    content_key = next((key for key in keys if key in batch[0][conversations_key][0].keys()), None)

    keys = ["prompt", "instruction"]
    prompt_key = next((key for key in keys if key in batch[0].keys()), None)
            
    conversations = [
        [{"role": role_dict[turn[role_key]], "content": turn[content_key]} for turn in sample[conversations_key]] 
        for sample in batch
    ]

    if prompt_key is not None and add_generation_prompt:
        conversations = [
            [{"role": "system", "content": sample[prompt_key]}] + conversation 
            for sample, conversation in zip(batch, conversations)
        ]
    
    # Tokenize and pad for the batch
    return tokenizer.apply_chat_template(
        conversation=conversations,
        **common_tokenizer_args,
        chat_template=chat_template,
        add_generation_prompt=add_generation_prompt,
        # tokenize=False,
        )

def collate_fn_instruction(batch, tokenizer, chat_template=None, add_generation_prompt=False, return_dict=False):
    """
    Collate function for conversation datasets.
    An instruction template is for single-turn tasks (user gives a task, model outputs a response).
    """
    conversations = []
    
    # Iterate over the batch of samples
    for sample in batch:

        # Context:
        keys = ["context", "input", "query", "problem"]
        user_key = next((key for key in keys if key in sample.keys()), None)

        # Response:
        keys = ["response", "output", "generated_solution", "solution"]
        assistant_key = next((key for key in keys if key in sample.keys()), None)

        # Append the conversation to the list
        conversations.append([
            {"role": "user", "content": sample.get('instruction', '') + sample.get(user_key, '')},
            {"role": "assistant", "content": sample.get(assistant_key, '')}
        ])


    return tokenizer.apply_chat_template(
        conversation=conversations,
        **common_tokenizer_args,
        chat_template=chat_template,
        add_generation_prompt=add_generation_prompt,
        )


def collate_fn_text(batch, tokenizer):
    """
    Collate function for text datasets.
    """

    for sample in batch:
        if "text" in sample.keys():
            continue
        for key in sample.keys():
            if key in ["data", "content"]:
                sample["text"] = sample[key]
                break
                # sample.pop(key)

    return tokenizer(
        [item["text"] for item in batch], 
        **common_tokenizer_args,
        )["input_ids"]


def collate_fn_preference(batch, tokenizer, chat_template=None, add_generation_prompt=False):
    """
    Collate function for preference datasets (DPO format).
    Handles different preference dataset formats:
    - Anthropic HH: {"chosen": [...], "rejected": [...]}
    - DPO datasets: {"prompt": "...", "chosen": "...", "rejected": "..."}
    - Custom: {"input": "...", "preferred": "...", "dispreferred": "..."}
    """
    prompts = []
    chosen_responses = []
    rejected_responses = []
    
    for sample in batch:
        # Find prompt (try multiple key names)
        prompt_keys = ["prompt", "input", "query", "context", "instruction"]
        prompt = None
        for key in prompt_keys:
            if key in sample:
                prompt = sample[key]
                break
        
        # If no prompt found, try to extract from chosen/rejected conversations
        if prompt is None:
            # Try to get prompt from chosen conversation
            if "chosen" in sample and isinstance(sample["chosen"], list):
                # Extract all non-assistant messages as prompt
                prompt_parts = [turn["content"] for turn in sample["chosen"] if turn.get("role") != "assistant"]
                prompt = "\n".join(prompt_parts) if prompt_parts else ""
        
        # Find chosen response
        chosen_keys = ["chosen", "preferred", "response", "output"]
        chosen = None
        for key in chosen_keys:
            if key in sample:
                chosen = sample[key]
                # Handle list format (Anthropic HH format)
                if isinstance(chosen, list):
                    # Extract assistant response from conversation
                    chosen = next((turn["content"] for turn in chosen if turn.get("role") == "assistant"), None)
                    if chosen is None and len(chosen) > 0:
                        # Fallback: use last message
                        chosen = chosen[-1].get("content", "")
                break
        
        # Find rejected response
        rejected_keys = ["rejected", "dispreferred"]
        rejected = None
        for key in rejected_keys:
            if key in sample:
                rejected = sample[key]
                # Handle list format
                if isinstance(rejected, list):
                    rejected = next((turn["content"] for turn in rejected if turn.get("role") == "assistant"), None)
                    if rejected is None and len(rejected) > 0:
                        # Fallback: use last message
                        rejected = rejected[-1].get("content", "")
                break
        
        if prompt is None:
            raise ValueError(f"Missing prompt in sample. Available keys: {list(sample.keys())}")
        if chosen is None:
            raise ValueError(f"Missing chosen response in sample. Available keys: {list(sample.keys())}")
        if rejected is None:
            raise ValueError(f"Missing rejected response in sample. Available keys: {list(sample.keys())}")
        
        prompts.append(prompt)
        chosen_responses.append(chosen)
        rejected_responses.append(rejected)
    
    # Tokenize prompts
    prompt_ids = tokenizer(
        prompts,
        **common_tokenizer_args,
        add_special_tokens=True,
    )["input_ids"]
    
    # Tokenize chosen responses
    chosen_ids = tokenizer(
        chosen_responses,
        **common_tokenizer_args,
        add_special_tokens=False,  # Don't add special tokens for responses
    )["input_ids"]
    
    # Tokenize rejected responses
    rejected_ids = tokenizer(
        rejected_responses,
        **common_tokenizer_args,
        add_special_tokens=False,
    )["input_ids"]
    
    return {
        "input_ids": prompt_ids,
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
    }
