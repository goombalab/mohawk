import random

import numpy
import torch
from huggingface_hub import HfApi
from contextlib import contextmanager

import time
from huggingface_hub.utils import HfHubHTTPError
import requests
from utils.logging import logger

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    numpy.random.seed(seed)
    random.seed(seed)

def nested_dict_to_table(data):
    # Function to flatten the dictionary
    def flatten_dict(d, parent_key='', sep='.'):
        if not isinstance(d, dict):
            return [(parent_key, d)]
        
        items = []
        for k, v in d.items():
            # Construct the new key
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, dict):
                # Recursively flatten the dictionary
                items.extend(flatten_dict(v, new_key, sep=sep))
            else:
                # Add the key-value pair to the items list
                items.append((new_key, v))
        return items

    # Flatten the dictionary
    flat_items = flatten_dict(data)

    # Determine column widths for formatting
    key_width = max(len(str(k)) for k, _ in flat_items + [('Key', '')])
    value_width = max(len(str(v)) for _, v in flat_items + [('', 'Value')])

    # Print the table header
    text = ""
    text += f"{'Key'.ljust(key_width)} | {'Value'.ljust(value_width)}\n"
    text += '-' * (key_width + value_width + 3) + "\n"

    # Print each key-value pair
    for key, value in flat_items:
        text += f"{key.ljust(key_width)} | {str(value).ljust(value_width)}\n"

    return text

# create a contextmanager to load the model weights with 3 tries
def recurrent_fn(n_tries=3):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            for _ in range(n_tries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    logger.error(f"[TRAINING_WRAPPER] Error saving state:\n{e}")
            return None
        return wrapper
    return decorator

def is_valid_hf_model_name(model_name: str) -> bool:
    """
    Check if a model name is valid for Hugging Face.

    Example usage
    model_name = "bert-base-uncased"
    assert is_valid_model_name(model_name), f"{model_name} is not a valid Hugging Face model name."
    """
    # try:
    #     # Try to retrieve model details
    #     HfApi().model_info(model_name)
    #     return True  # If no error occurs, the model name is valid
    # except Exception as e:
    #     logger.warning(f"[IS_VALID_HF_MODEL_NAME] Error: \n{e}\n The model name is not valid.")
    #     return False  # If an error occurs, the model name is not valid
    logger.warning(f"[IS_VALID_HF_MODEL_NAME] A temporary fix for the Hugging Face model name validation.")
    return True


@contextmanager
def catch_huggingface_http_errors(retries=10, delay=5, raise_on_failure=True):
    attempt = 0
    while True:
        try:
            yield
            break  # Exit the loop if successful
        except (HfHubHTTPError, requests.exceptions.HTTPError) as e:
            attempt += 1
            logger.info(f"[Attempt {attempt}/{retries}] Error: \n{e}\n Retrying in {delay} seconds...")
            if attempt >= retries and raise_on_failure:
                logger.info(f"[Attempt {attempt}/{retries}] Error: \n{e}\n Max retries reached. Exiting...")
                raise
            elif attempt >= retries and not raise_on_failure:
                logger.info(f"[Attempt {attempt}/{retries}] Error: \n{e}\n Max retries reached. Continuing...")
                break
            time.sleep(delay)
