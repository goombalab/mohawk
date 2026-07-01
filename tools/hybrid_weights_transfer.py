import os
import argparse
import sys
from pathlib import Path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Transfer selected teacher attention heads into a hybrid student model."
    )
    parser.add_argument(
        "--config",
        default="./configs/Qwen2/1.5B/hybrid/architecture_5.yaml",
        help="Path to the hybrid model config.",
    )
    parser.add_argument(
        "--heads",
        default=None,
        help=(
            "Comma-separated layer:head pairs to transfer, for example '0:0,1:2'. "
            "When omitted, the built-in production teacher map is used."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Device for the smoke/probe forward. Defaults to CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--allow-unexpected-student-load",
        action="store_true",
        help="Allow unexpected teacher keys while preloading the student from teacher weights.",
    )
    return parser


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    build_arg_parser().print_help()
    raise SystemExit(0)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from utils.logging import init_logging
from utils.init_model import eager_init
from utils.config import load_config, Config

os.environ["LOGURU_LEVEL"] = "DEBUG"


HEADS_BY_TEACHER = {
    "meta-llama/Llama-3.2-1B-Instruct": [
        (9, 25), (4, 4), (10, 26), (0, 20), (8, 20), (8, 19), (8, 16), (7, 22), (10, 20), (5, 5),
        (7, 13), (11, 8), (1, 28), (4, 26), (5, 18), (5, 31), (0, 7), (3, 29), (10, 9), (6, 31),
        (8, 21), (5, 27), (6, 15), (15, 14), (8, 30), (9, 12), (3, 8), (4, 6), (5, 4), (0, 11),
        (3, 15), (5, 1), (5, 13), (4, 11), (12, 3), (0, 3), (3, 16), (5, 17), (7, 24), (8, 5),
    ],
    "meta-llama/Llama-3.1-8B-Instruct": [
        (13, 18), (15, 11), (17, 24), (13, 17), (13, 16), (14, 21), (15, 8), (13, 0), (15, 9), (17, 27),
        (14, 30), (21, 10), (16, 16), (12, 20), (16, 22), (14, 28), (14, 16), (16, 0), (13, 13), (16, 28),
        (13, 24), (13, 10), (11, 5), (9, 10), (12, 0), (7, 17), (15, 0), (12, 3), (7, 6), (15, 6),
        (13, 28), (2, 30), (9, 18), (15, 18), (23, 14), (0, 2), (7, 23), (12, 25), (22, 13), (11, 4),
    ],
    "Qwen/Qwen2.5-1.5B-Instruct": [
        (13, 4), (18, 0), (16, 3), (16, 6), (22, 7), (18, 1), (16, 1), (17, 2), (21, 11), (15, 10),
        (18, 5), (17, 5), (26, 9), (15, 7), (14, 2), (16, 2), (14, 10), (11, 9), (8, 9), (13, 5),
        (5, 1), (12, 3), (5, 11), (19, 9), (16, 5), (20, 0), (27, 5), (17, 10), (15, 3), (10, 11),
        (10, 2), (19, 4), (13, 9), (11, 5), (13, 8), (1, 5), (11, 0), (0, 11), (8, 6), (3, 2),
    ],
}


def parse_args():
    return build_arg_parser().parse_args()


def get_heads_for_teacher(teacher_dir):
    if teacher_dir not in HEADS_BY_TEACHER:
        supported = ", ".join(HEADS_BY_TEACHER.keys())
        raise ValueError(f"Teacher model {teacher_dir} not supported. Supported models: {supported}")
    return HEADS_BY_TEACHER[teacher_dir]


def parse_heads(heads_arg):
    if heads_arg is None:
        return None
    heads = []
    for item in heads_arg.split(","):
        layer_idx, head_idx = item.split(":", 1)
        heads.append((int(layer_idx), int(head_idx)))
    return heads


def apply_config_env_vars(cfg):
    env_vars = getattr(getattr(cfg, "ManagementConfig", None), "env_vars", {})
    for key, value in getattr(env_vars, "items", lambda: [])():
        if value is not None:
            os.environ[str(key)] = str(value)


def resolve_device(device_arg):
    if device_arg is not None:
        if device_arg == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return torch.device("cpu")
        if device_arg == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is unavailable.")
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_from_config(cfg):
    seed = getattr(getattr(cfg, "TrainConfig", None), "seed", None)
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def main():
    args = parse_args()

    cfg = load_config(
        config_path=args.config,
        CONSTANTS={"slurm_job_id": 0},
    )
    apply_config_env_vars(cfg)
    device = resolve_device(args.device)
    seed_from_config(cfg)

    cfg.LoadConfig = Config.from_dict({
        "model": [
            {
                "allow_missing_keys": True,
                "allow_unexpected_keys": args.allow_unexpected_student_load,
                "path": cfg.TeacherConfig.dir,
                "rename": {
                    ".norm.": ".final_layernorm.",
                    "embed_tokens.": "embedding.",
                    "model.": "backbone.",
                    "module.": "",
                },
                "black_list": [
                    "self_attn",
                    "lm_head",
                ],
            }
        ]
    })

    heads = parse_heads(args.heads)
    if heads is None:
        heads = get_heads_for_teacher(cfg.TeacherConfig.dir)

    with init_logging(cfg, wandb_id="0"):
        student_wrapper = eager_init(
            components_cfg=cfg.ComponentsConfig,
            details_cfg=cfg.TrainConfig,
            load_cfg=cfg.LoadConfig,
            cfg=cfg,
            mode="inference",
        )
        student_wrapper(input_ids=torch.zeros((1, 1), dtype=torch.int64, device=device))

        teacher_wrapper = eager_init(
            details_cfg=cfg.TeacherConfig,
            load_cfg=Config.from_dict({"model": [{"path": cfg.TeacherConfig.dir}]}),
            cfg=cfg,
            mode="inference",
        )

    num_heads = sum(
        layer.mixer2.self_attn.config.num_attention_heads
        for layer in student_wrapper.model.backbone.layers
        if hasattr(layer, "mixer2")
    )
    heads = heads[:num_heads]
    print(f"Using config: {args.config}")
    print(f"Teacher: {cfg.TeacherConfig.dir}")
    print(f"Transferring {len(heads)} attention heads to student model out of {num_heads} available heads.")

    heads_dict = {
        layer_idx: [h for l, h in heads if l == layer_idx]
        for layer_idx in range(cfg.ComponentsConfig.MixerModel.input.n_layer)
    }
    heads_dict = {layer_idx: heads for layer_idx, heads in heads_dict.items() if len(heads) > 0}

    for layer_idx, heads in heads_dict.items():
        assert student_wrapper.model.backbone.layers[layer_idx].mixer2.self_attn.config.num_attention_heads == len(heads)

        teacher_self_attn = teacher_wrapper.model.model.layers[layer_idx].self_attn
        teacher_num_heads = teacher_self_attn.config.num_attention_heads
        teacher_head_dim = getattr(teacher_self_attn.config, "head_dim", None) or getattr(teacher_self_attn, "head_dim", None)
        teacher_num_kv_heads = teacher_self_attn.config.num_key_value_heads
        teacher_hidden_size = teacher_self_attn.config.hidden_size
        repeat_factor = teacher_num_heads // teacher_num_kv_heads

        student_head_dim = student_wrapper.model.backbone.layers[layer_idx].mixer2.self_attn.config.head_dim
        assert teacher_head_dim == student_head_dim, (
            f"Teacher head dim {teacher_head_dim} does not match student head dim {student_head_dim}"
        )

        teacher_weights = teacher_wrapper.model.model.layers[layer_idx].self_attn.q_proj.weight.data
        teacher_weights = teacher_weights.view(teacher_num_heads, teacher_head_dim, teacher_hidden_size)
        student_wrapper.model.backbone.layers[layer_idx].mixer2.self_attn.q_proj.weight.data = (
            teacher_weights[heads].view(-1, teacher_hidden_size).contiguous().clone()
        )

        if student_wrapper.model.backbone.layers[layer_idx].mixer2.self_attn.q_proj.bias is not None:
            teacher_bias = teacher_wrapper.model.model.layers[layer_idx].self_attn.q_proj.bias.data
            teacher_bias = teacher_bias.view(teacher_num_heads, teacher_head_dim)
            student_wrapper.model.backbone.layers[layer_idx].mixer2.self_attn.q_proj.bias.data = (
                teacher_bias[heads].view(-1).contiguous().clone()
            )

        teacher_weights = teacher_wrapper.model.model.layers[layer_idx].self_attn.k_proj.weight.data
        teacher_weights = teacher_weights.view(teacher_num_kv_heads, teacher_head_dim, teacher_hidden_size)
        teacher_weights = teacher_weights.repeat_interleave(repeat_factor, dim=0)
        student_wrapper.model.backbone.layers[layer_idx].mixer2.self_attn.k_proj.weight.data = (
            teacher_weights[heads].view(-1, teacher_hidden_size).contiguous().clone()
        )

        if student_wrapper.model.backbone.layers[layer_idx].mixer2.self_attn.k_proj.bias is not None:
            teacher_bias = teacher_wrapper.model.model.layers[layer_idx].self_attn.k_proj.bias.data
            teacher_bias = teacher_bias.view(teacher_num_kv_heads, teacher_head_dim)
            teacher_bias = teacher_bias.repeat_interleave(repeat_factor, dim=0)
            student_wrapper.model.backbone.layers[layer_idx].mixer2.self_attn.k_proj.bias.data = (
                teacher_bias[heads].view(-1).contiguous().clone()
            )

        teacher_weights = teacher_wrapper.model.model.layers[layer_idx].self_attn.v_proj.weight.data
        teacher_weights = teacher_weights.view(teacher_num_kv_heads, teacher_head_dim, teacher_hidden_size)
        teacher_weights = teacher_weights.repeat_interleave(repeat_factor, dim=0)
        student_wrapper.model.backbone.layers[layer_idx].mixer2.self_attn.v_proj.weight.data = (
            teacher_weights[heads].view(-1, teacher_hidden_size).contiguous().clone()
        )

        if student_wrapper.model.backbone.layers[layer_idx].mixer2.self_attn.v_proj.bias is not None:
            teacher_bias = teacher_wrapper.model.model.layers[layer_idx].self_attn.v_proj.bias.data
            teacher_bias = teacher_bias.view(teacher_num_kv_heads, teacher_head_dim)
            teacher_bias = teacher_bias.repeat_interleave(repeat_factor, dim=0)
            student_wrapper.model.backbone.layers[layer_idx].mixer2.self_attn.v_proj.bias.data = (
                teacher_bias[heads].view(-1).contiguous().clone()
            )

        teacher_weights = teacher_wrapper.model.model.layers[layer_idx].self_attn.o_proj.weight.data
        teacher_weights = teacher_weights.view(teacher_hidden_size, teacher_num_heads, teacher_head_dim)
        student_wrapper.model.backbone.layers[layer_idx].mixer2.self_attn.o_proj.weight.data = (
            teacher_weights[:, heads].view(teacher_hidden_size, -1).contiguous().clone()
        )

        if student_wrapper.model.backbone.layers[layer_idx].mixer2.self_attn.o_proj.bias is not None:
            teacher_bias = teacher_wrapper.model.model.layers[layer_idx].self_attn.o_proj.bias.data
            student_wrapper.model.backbone.layers[layer_idx].mixer2.self_attn.o_proj.bias.data = (
                teacher_bias[heads].contiguous().clone()
            )

        print(f"Transferred heads {heads} to student layer {layer_idx}.")

    student_wrapper.save_weights()
    print("Done.")


if __name__ == "__main__":
    main()
