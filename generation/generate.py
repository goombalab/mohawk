# Copyright (c) 2024, Aviv Bick, Kevin Li.

import argparse
import contextlib
import sys
import time
from functools import partial
from pathlib import Path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate text from a supported local or Hugging Face causal LM.")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--promptlen", type=int, default=200)
    parser.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--genlen", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument(
        "--model_dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype to use. 'auto' keeps a local Mohawk checkpoint's saved dtype when available.",
    )
    # Sampling arguments
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--local_files_only", action="store_true", help="Only use locally cached model/tokenizer files.")
    parser.add_argument(
        "--model-registration-module",
        action="append",
        default=[],
        metavar="MODULE",
        help="Import MODULE before model loading so it can register a custom Transformers model type. Repeat as needed.",
    )
    return parser


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    build_arg_parser().print_help()
    raise SystemExit(0)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from utils.build_model import (
    build_n_load_model,
    get_tokenizer,
    import_model_registration_modules,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    return build_arg_parser().parse_args()


def _maybe_cuda_synchronize():
    if device == "cuda":
        torch.cuda.synchronize()


def _autocast_context(enabled):
    if device != "cuda":
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type="cuda", enabled=enabled, dtype=torch.bfloat16)


@torch.inference_mode()
def time_bench(args, input_ids, generate_fn):
    _maybe_cuda_synchronize()
    start = time.time()
    with _autocast_context(args.mixed_precision):
        for _ in range(args.repeats):
            out = generate_fn(
                input_ids=input_ids, max_length=input_ids.shape[1] + args.genlen
            )
    _maybe_cuda_synchronize()

    # Print stats
    print(f"\nTiming results for {args.model} model:")
    print(
        f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}"
    )
    print(
        f"prompt processing + decoding time: {(time.time() - start) / args.repeats * 1000:.0f}ms"
    )


@torch.inference_mode()
def main():

    # Parse arguments
    args = parse_args()
    torch.manual_seed(args.seed)
    import_model_registration_modules(args.model_registration_module)

    
    # Load model
    model = build_n_load_model(
        args.model,
        model_dtype=args.model_dtype,
        local_files_only=args.local_files_only,
    )
    tokenizer = get_tokenizer(args.model, local_files_only=args.local_files_only)
    model = model.model

    # Prepare model
    model.to(device=device)
    model.eval()
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    allowed_models = ["mamba", "31559"]
    unallowed_models = ["falcon"]
    has_attention_mixer = any(hasattr(module, "mixer2") for module in model.modules())
    optimizable = (
        any(model_name in args.model for model_name in allowed_models)
        and not any(model_name in args.model for model_name in unallowed_models)
        and not has_attention_mixer
    )

    # Tokenize prompt
    if args.prompt is None:
        input_ids = torch.randint(
            1, 1000, (1, args.promptlen), dtype=torch.long, device=device
        )
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    else:
        tokens = tokenizer(args.prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(device=device)
        attn_mask = tokens.attention_mask.to(device=device)


    generate_fn = partial(
        model.generate,
        return_dict_in_generate=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        output_scores=False,
        # HF GenerationMixin
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        attention_mask=attn_mask,
        )
    
    if optimizable and device == "cuda":
        generate_fn = partial(
            generate_fn,
            # Mamba GenerationMixin
            cg=True,
            enable_timing=False,
            eos_token_id=tokenizer.eos_token_id,
            )

    # Generation example
    with _autocast_context(args.mixed_precision):
        out = generate_fn(
            input_ids=input_ids, max_length=input_ids.shape[1] + args.genlen
        )

    if args.prompt is not None:
        print(
            "Generated text:\n",
            tokenizer.batch_decode(
                sequences=out.sequences.tolist(), 
                skip_special_tokens=True
            )[0],
        )

    time_bench(args, input_ids, generate_fn)


if __name__ == "__main__":
    main()
