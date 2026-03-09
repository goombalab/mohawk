import time
import torch
from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel
from types import SimpleNamespace
import traceback
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

device = 'cuda'

def mistarl_to_mamba():
    mistral_cfg = AutoConfig.from_pretrained("mistralai/Mamba-Codestral-7B-v0.1")
    
    config = MambaConfig()
    config.d_model = mistral_cfg.hidden_size
    config.n_layer = mistral_cfg.num_hidden_layers
    config.vocab_size = 50277
    config.ssm_cfg = {
        "layer": "Mamba2",
        "d_state": 128,
        "d_conv": 4,
        "conv_init": None,
        "expand": 2,
        "headdim": 64,
        "d_ssm": None,
        "ngroups": 8,
        "A_init_range": (1, 16),
        "D_has_hdim": False,
        "rmsnorm": True,
        "norm_before_gate": False,
        "dt_min": 0.001,
        "dt_max": 0.1,
        "dt_init_floor": 1e-4,
        "dt_limit": (0.0, float("inf")),
        "bias": False,
        "conv_bias": True,
        "chunk_size": 256,
        "use_mem_eff_path": True,
        "process_group": None,
        "sequence_parallel": True,
    }
    config._name_or_path = mistral_cfg._name_or_path
    return MambaLMHeadModel(config)

def falcon_to_mamba():
    falcon_cfg = AutoConfig.from_pretrained("tiiuae/falcon-mamba-7b")
    
    config = MambaConfig()
    config.d_model = falcon_cfg.hidden_size
    config.n_layer = falcon_cfg.num_hidden_layers
    config.vocab_size = 65024
    config.ssm_cfg = {
        "layer": "Mamba1",
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "bias": False,
        "conv_bias": True,
    }
    config._name_or_path = falcon_cfg._name_or_path
    return MambaLMHeadModel(config)

def load_model(name='llamba'):
    if name == 'llama':
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    elif name == 'codestral':
        # model = AutoModelForCausalLM.from_pretrained('mistralai/Mamba-Codestral-7B-v0.1')
        model = mistarl_to_mamba()
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mamba-Codestral-7B-v0.1')
    elif name == 'falcon':
        # model = AutoModelForCausalLM.from_pretrained('tiiuae/falcon-mamba-7b-instruct')
        model = falcon_to_mamba()
        tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-mamba-7b-instruct')
    elif name == 'llamba':
        model = LlambaLMHeadModel.from_pretrained('cartesia-ai/Llamba-8B', strict=True)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    else:
        raise ValueError(f"Model {name} not supported")

    # Additionals
    model = model.to(torch.bfloat16).to(device)
    model.requires_grad_(False).eval()
    # model = torch.compile(model)

    # print parameters count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters count: {num_params / 1e9:.2f}B")
    # print model size in GB:
    model_size = sum(p.element_size() * p.numel() for p in model.parameters()) / 1e9
    print(f"Model size: {model_size:.2f}GB")

    return model, tokenizer

@torch.inference_mode()
def benchmark_generation_throughput(model, tokenizer, batch_size: int, max_gen_len: int, prompt_len: int = None):
    """
    Benchmarks model generation throughput using CUDA Graphs.

    Args:
        model: The model instance.
        batch_size (int): Number of sequences.
        prompt_len (int): Length of the prompt.
        max_gen_len (int): Maximum number of tokens to generate.

    Returns:
        float: Throughput in tokens per second.
    """
    # Fill the first `prompt_len` tokens with random integers
    if prompt_len is None:
        dummy_input = tokenizer.encode("Hello, my dog is", return_tensors="pt")
    else:
        dummy_input = torch.randint(low=0, high=model.config.vocab_size, size=(batch_size, prompt_len), device=device)

    # Create static input buffer
    prompt_len = dummy_input.shape[1]
    max_length = prompt_len + max_gen_len

    dummy_input = dummy_input.expand(batch_size, prompt_len)

    static_inputs = {"input_ids": dummy_input}

    # Initialize cache \ state
    if isinstance(model, LlambaLMHeadModel) or isinstance(model, MambaLMHeadModel):
        state = SimpleNamespace()
        state.key_value_memory_dict = model.allocate_inference_cache(batch_size, max_length, dtype=torch.bfloat16)
        state.batch_size = batch_size
        state.seqlen_offset = 0
        static_inputs["inference_params"] = state
    elif isinstance(model, LlamaForCausalLM):
        static_inputs["use_cache"] = True
        static_inputs["past_key_values"] = StaticCache(config=model.config, max_batch_size=batch_size, max_cache_len=max_length, device=device, dtype=torch.bfloat16)
    else:
        raise ValueError(f"Model {model.__class__.__name__} not supported")
    
    # Pre-fill the cache \ state in-place
    static_outputs = model(**static_inputs) # inplace update
    next_token = static_outputs.logits[:, -1].argmax(dim=-1, keepdim=True)

    # Update the input buffer with the first generated token
    static_inputs["input_ids"] = next_token

    # Update the current position in the cache \ state
    if 'past_key_values' in static_inputs:
        static_inputs["cache_position"] = torch.arange(prompt_len, prompt_len + static_inputs["input_ids"].shape[1], device=device)
    if 'inference_params' in static_inputs:
        static_inputs["inference_params"].seqlen_offset = prompt_len  # Start at prompt length

    # Warm-up kernel (to avoid one-time setup overhead)
    n_warmups = 2
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_outputs = model(**static_inputs)
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    # Capture CUDA Graph for a **single-step** inference
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        try:
            static_outputs = model(**static_inputs)
        except Exception as e:
            traceback.print_exc()

    # Measure throughput
    torch.cuda.synchronize()

    return_tokens = False
    if return_tokens:
        generated_tokens = static_inputs["input_ids"].clone()
    timer = 0
    print(f"Batch size; Generated tokens; Throughput (tokens/s); Allocated memory (GB)")
    # for i in range(prompt_len, max_length):
    for i in range(1, max_gen_len + 1):
        # Extract the last generated token and set it as the next input
        static_inputs["input_ids"].copy_(next_token)

        # Run CUDA Graph inference
        start_time = time.time()
        graph.replay()
        next_token = static_outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        timer += time.time() - start_time

        # Append the new token to the generated sequence
        if return_tokens:
            generated_tokens = torch.cat((generated_tokens, next_token), dim=-1)
        if 'past_key_values' in static_inputs:
            static_inputs["cache_position"] += 1
        if 'inference_params' in static_inputs:
            static_inputs['inference_params'].seqlen_offset += 1

        # Time
        if i & (i - 1) == 0:
            # throughput = (batch_size * (i-prompt_len)) / timer
            throughput = (batch_size * i) / timer
            print(f"{batch_size}; {i}; {throughput:.2f}; {torch.cuda.memory_allocated() / 1e9:.2f}")

    if return_tokens:
        assert generated_tokens.shape == (batch_size, max_length)
    if 'past_key_values' in static_inputs:
        assert (static_inputs["past_key_values"].value_cache[0][:,:,-1,:] != 0).any(), "Last KV is not filled, which is weird"
        # value_cache[0][:,:,i,:] should be filled, value_cache[0][:,:,i+1,:] should be zeros
    if 'inference_params' in static_inputs:
        assert static_inputs['inference_params'].seqlen_offset == max_length, "seqlen_offset is not max_length"

    return throughput

if __name__ == "__main__":

    model, tokenizer = load_model()

    # Example parameters – adjust these as needed:
    batch_size = 32   # Number of sequences in the batch
    prompt_len = 1  # Length of the prompt (in tokens)
    max_gen_len = 4096  # Number of tokens to generate (2048, 4096, 8192)

    print(f"Model: {model.config._name_or_path}, Prompt length: {prompt_len}, Max Generation length: {max_gen_len}")
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        try:
            benchmark_generation_throughput(model, tokenizer, batch_size, max_gen_len, prompt_len)
        except Exception as e:
            if "CUDA out of memory" in str(e):
                print(f"Batch size: {batch_size} | CUDA out of memory")
            else:
                traceback.print_exc()
            break
