import time
import argparse
import json
import sys


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Research CUDA-graph throughput benchmark for selected Llama/Mamba-family models."
    )
    parser.add_argument("--model", default="llamba", choices=["llama", "codestral", "falcon", "llamba", "hf"])
    parser.add_argument(
        "--hf-model-name",
        default="hf-internal-testing/tiny-random-LlamaForCausalLM",
        help="Public Hugging Face causal LM to load when --model hf is selected.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load all Hugging Face config, model, and tokenizer files from the local cache only.",
    )
    parser.add_argument(
        "--llamba-config-only",
        action="store_true",
        help=(
            "Construct the Llamba-8B architecture from its config with random "
            "weights, without downloading checkpoint weights or a tokenizer."
        ),
    )
    parser.add_argument("--prompt-len", type=int, default=1)
    parser.add_argument("--max-gen-len", type=int, default=4096)
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096",
        help="Comma-separated batch sizes to try.",
    )
    return parser


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    build_arg_parser().print_help()
    raise SystemExit(0)


import torch
from types import SimpleNamespace
import traceback

LlambaLMHeadModel = None
LlamaForCausalLM = None
MambaLMHeadModel = None
StaticCache = None

device = 'cuda'


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "tools/benchmark_throughput.py requires CUDA because it benchmarks "
            "CUDA graph generation throughput."
        )


def get_transformers():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    return AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_name, local_files_only=False):
    _, _, AutoTokenizer = get_transformers()
    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
    except ValueError as exc:
        if "sentencepiece" not in str(exc).lower():
            raise

        from transformers import PreTrainedTokenizerFast
        from transformers.utils.hub import cached_file

        tokenizer_file = cached_file(
            model_name,
            "tokenizer.json",
            local_files_only=local_files_only,
        )
        if tokenizer_file is None:
            raise
        print(
            f"AutoTokenizer requires SentencePiece for {model_name}; "
            "using its tokenizer.json fast-tokenizer artifact for throughput",
            flush=True,
        )
        return PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)


def get_llama_model_cls():
    global LlamaForCausalLM
    if LlamaForCausalLM is None:
        from transformers.models.llama.modeling_llama import LlamaForCausalLM as _LlamaForCausalLM

        LlamaForCausalLM = _LlamaForCausalLM
    return LlamaForCausalLM


def get_static_cache_cls():
    global StaticCache
    if StaticCache is None:
        from transformers.cache_utils import StaticCache as _StaticCache

        StaticCache = _StaticCache
    return StaticCache


def get_mamba_classes():
    global MambaLMHeadModel
    if MambaLMHeadModel is None:
        from mamba_ssm import MambaLMHeadModel as _MambaLMHeadModel

        MambaLMHeadModel = _MambaLMHeadModel
    from mamba_ssm.models.config_mamba import MambaConfig

    return MambaLMHeadModel, MambaConfig


def get_llamba_model_cls():
    global LlambaLMHeadModel
    if LlambaLMHeadModel is None:
        from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel as _LlambaLMHeadModel

        LlambaLMHeadModel = _LlambaLMHeadModel
    return LlambaLMHeadModel

def mistarl_to_mamba(local_files_only=False):
    AutoConfig, _, _ = get_transformers()
    MambaLMHeadModel, MambaConfig = get_mamba_classes()
    print("Loading Codestral production config", flush=True)
    mistral_cfg = AutoConfig.from_pretrained(
        "mistralai/Mamba-Codestral-7B-v0.1",
        local_files_only=local_files_only,
    )
    
    config = MambaConfig()
    config.d_model = mistral_cfg.hidden_size
    config.n_layer = mistral_cfg.num_hidden_layers
    config.vocab_size = mistral_cfg.vocab_size
    config.ssm_cfg = {
        "layer": "Mamba2",
        "d_state": mistral_cfg.state_size,
        "d_conv": mistral_cfg.conv_kernel,
        "conv_init": None,
        "expand": mistral_cfg.intermediate_size // mistral_cfg.hidden_size,
        "headdim": mistral_cfg.head_dim,
        "d_ssm": None,
        "ngroups": mistral_cfg.n_groups,
        "A_init_range": (1, 16),
        "D_has_hdim": False,
        "rmsnorm": mistral_cfg.rms_norm,
        "norm_before_gate": mistral_cfg.norm_before_gate,
        "dt_min": mistral_cfg.time_step_min,
        "dt_max": mistral_cfg.time_step_max,
        "dt_init_floor": mistral_cfg.time_step_floor,
        "dt_limit": (0.0, float("inf")),
        "bias": False,
        "conv_bias": True,
        "chunk_size": 256,
        "use_mem_eff_path": True,
        "process_group": None,
        "sequence_parallel": True,
    }
    config._name_or_path = mistral_cfg._name_or_path
    print(
        f"Constructing Codestral-shaped Mamba model: "
        f"d_model={config.d_model}, n_layer={config.n_layer}, "
        f"vocab_size={config.vocab_size}",
        flush=True,
    )
    return MambaLMHeadModel(config)

def falcon_to_mamba(local_files_only=False):
    AutoConfig, _, _ = get_transformers()
    MambaLMHeadModel, MambaConfig = get_mamba_classes()
    print("Loading Falcon-Mamba production config", flush=True)
    falcon_cfg = AutoConfig.from_pretrained(
        "tiiuae/falcon-mamba-7b",
        local_files_only=local_files_only,
    )
    
    config = MambaConfig()
    config.d_model = falcon_cfg.hidden_size
    config.n_layer = falcon_cfg.num_hidden_layers
    config.vocab_size = falcon_cfg.vocab_size
    config.ssm_cfg = {
        "layer": "Mamba1",
        "d_state": falcon_cfg.state_size,
        "d_conv": falcon_cfg.conv_kernel,
        "expand": falcon_cfg.intermediate_size // falcon_cfg.hidden_size,
        "bias": falcon_cfg.use_bias,
        "conv_bias": falcon_cfg.use_conv_bias,
    }
    config._name_or_path = falcon_cfg._name_or_path
    print(
        f"Constructing Falcon-shaped Mamba model: "
        f"d_model={config.d_model}, n_layer={config.n_layer}, "
        f"vocab_size={config.vocab_size}",
        flush=True,
    )
    return MambaLMHeadModel(config)

def load_model(
    name='llamba',
    hf_model_name=None,
    local_files_only=False,
    llamba_config_only=False,
):
    _, AutoModelForCausalLM, _ = get_transformers()
    if name == 'llama':
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )
        tokenizer = load_tokenizer(
            "meta-llama/Llama-3.1-8B-Instruct",
            local_files_only=local_files_only,
        )
    elif name == 'hf':
        if hf_model_name is None:
            raise ValueError("--hf-model-name is required when --model hf is selected")
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )
        tokenizer = load_tokenizer(
            hf_model_name,
            local_files_only=local_files_only,
        )
    elif name == 'codestral':
        # model = AutoModelForCausalLM.from_pretrained('mistralai/Mamba-Codestral-7B-v0.1')
        started = time.perf_counter()
        model = mistarl_to_mamba(local_files_only=local_files_only)
        print(
            f"Codestral-shaped model constructed in {time.perf_counter() - started:.2f}s",
            flush=True,
        )
        tokenizer = load_tokenizer(
            'mistralai/Mamba-Codestral-7B-v0.1',
            local_files_only=local_files_only,
        )
    elif name == 'falcon':
        # model = AutoModelForCausalLM.from_pretrained('tiiuae/falcon-mamba-7b-instruct')
        started = time.perf_counter()
        model = falcon_to_mamba(local_files_only=local_files_only)
        print(
            f"Falcon-shaped model constructed in {time.perf_counter() - started:.2f}s",
            flush=True,
        )
        tokenizer = load_tokenizer(
            'tiiuae/falcon-mamba-7b-instruct',
            local_files_only=local_files_only,
        )
    elif name == 'llamba':
        LlambaLMHeadModel = get_llamba_model_cls()
        if llamba_config_only:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id="cartesia-ai/Llamba-8B",
                filename="config.json",
                local_files_only=local_files_only,
            )
            with open(config_path) as config_file:
                config = json.load(config_file)
            print(
                "Constructing Llamba-8B from production config with random weights: "
                f"d_model={config['d_model']}, n_layer={config['n_layer']}, "
                f"vocab_size={config['vocab_size']}",
                flush=True,
            )
            model = LlambaLMHeadModel(config)
            model.config._name_or_path = "cartesia-ai/Llamba-8B (config-only)"
            tokenizer = None
        else:
            model = LlambaLMHeadModel.from_pretrained('cartesia-ai/Llamba-8B', strict=True)
            tokenizer = load_tokenizer(
                "meta-llama/Llama-3.1-8B-Instruct",
                local_files_only=local_files_only,
            )
    else:
        raise ValueError(f"Model {name} not supported")

    if name in {"codestral", "falcon"}:
        if tokenizer.vocab_size != model.config.vocab_size:
            raise ValueError(
                f"{name} tokenizer/model vocabulary mismatch: "
                f"{tokenizer.vocab_size} != {model.config.vocab_size}"
            )
        print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}", flush=True)

    # Additionals
    print("Moving model to bfloat16 CUDA", flush=True)
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
        if tokenizer is None:
            raise ValueError("A tokenizer is required when prompt_len is not set")
        dummy_input = tokenizer.encode("Hello, my dog is", return_tensors="pt")
    else:
        dummy_input = torch.randint(low=0, high=model.config.vocab_size, size=(batch_size, prompt_len), device=device)

    # Create static input buffer
    prompt_len = dummy_input.shape[1]
    max_length = prompt_len + max_gen_len

    dummy_input = dummy_input.expand(batch_size, prompt_len)

    static_inputs = {"input_ids": dummy_input}

    # Initialize cache \ state
    llamba_cls = LlambaLMHeadModel
    mamba_cls = MambaLMHeadModel
    if (llamba_cls is not None and isinstance(model, llamba_cls)) or (
        mamba_cls is not None and isinstance(model, mamba_cls)
    ):
        state = SimpleNamespace()
        state.key_value_memory_dict = model.allocate_inference_cache(batch_size, max_length, dtype=torch.bfloat16)
        state.batch_size = batch_size
        state.seqlen_offset = 0
        static_inputs["inference_params"] = state
        static_inputs["num_last_tokens"] = 1
    elif isinstance(model, get_llama_model_cls()):
        static_inputs["use_cache"] = True
        static_inputs["past_key_values"] = get_static_cache_cls()(config=model.config, max_batch_size=batch_size, max_cache_len=max_length, device=device, dtype=torch.bfloat16)
    else:
        raise ValueError(f"Model {model.__class__.__name__} not supported")
    
    # Pre-fill the cache \ state in-place
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    prefill_start.record()
    static_outputs = model(**static_inputs)  # inplace update
    prefill_end.record()
    prefill_end.synchronize()
    prefill_seconds = prefill_start.elapsed_time(prefill_end) / 1000.0
    prefill_tokens = batch_size * prompt_len
    print(
        "Prefill batch; Prompt tokens; Time (s); Throughput (tokens/s); "
        "Allocated memory (GB)"
    )
    print(
        f"{batch_size}; {prompt_len}; {prefill_seconds:.6f}; "
        f"{prefill_tokens / prefill_seconds:.2f}; "
        f"{torch.cuda.memory_allocated() / 1e9:.2f}"
    )
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
    timer = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    print(f"Batch size; Generated tokens; Throughput (tokens/s); Allocated memory (GB)")
    # for i in range(prompt_len, max_length):
    for i in range(1, max_gen_len + 1):
        # Extract the last generated token and set it as the next input
        static_inputs["input_ids"].copy_(next_token)

        # Run CUDA Graph inference
        graph.replay()
        next_token = static_outputs.logits[:, -1].argmax(dim=-1, keepdim=True)

        # Append the new token to the generated sequence
        if return_tokens:
            generated_tokens = torch.cat((generated_tokens, next_token), dim=-1)
        if 'past_key_values' in static_inputs:
            static_inputs["cache_position"] += 1
        if 'inference_params' in static_inputs:
            static_inputs['inference_params'].seqlen_offset += 1

        # Time
        if i & (i - 1) == 0:
            end_event.record()
            end_event.synchronize()
            timer = start_event.elapsed_time(end_event) / 1000.0
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

    parser = build_arg_parser()
    args = parser.parse_args()
    if args.llamba_config_only and args.model != "llamba":
        parser.error("--llamba-config-only requires --model llamba")
    require_cuda()
    model, tokenizer = load_model(
        args.model,
        hf_model_name=args.hf_model_name,
        local_files_only=args.local_files_only,
        llamba_config_only=args.llamba_config_only,
    )

    # Example parameters – adjust these as needed:
    prompt_len = args.prompt_len
    max_gen_len = args.max_gen_len

    print(f"Model: {model.config._name_or_path}, Prompt length: {prompt_len}, Max Generation length: {max_gen_len}")
    for batch_size in [int(item) for item in args.batch_sizes.split(",") if item.strip()]:
        try:
            benchmark_generation_throughput(model, tokenizer, batch_size, max_gen_len, prompt_len)
        except Exception as e:
            if "CUDA out of memory" in str(e):
                print(f"Batch size: {batch_size} | CUDA out of memory")
            else:
                traceback.print_exc()
            break
