import tempfile

import pytest

pytestmark = [pytest.mark.slow]

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from utils.build_model import build_local_model
from utils.config import Config


def _base_input():
    return {
        "input_size": 32,
        "n_layer": 1,
        "d_model": 16,
        "norm_epsilon": 1e-5,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 16,
        "rope_theta": 10000.0,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "_attn_implementation": "eager",
    }


def _attention_layer(name):
    return {
        "name": name,
        "input": {
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 16,
            "rope_theta": 10000.0,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "_attn_implementation": "eager",
        },
    }


def _family_config(family):
    mixer_name, block_name, layer = family
    return Config.from_dict(
        {
            "name": "LayeredMambaLM",
            "input": {
                "vocab_size": 32,
                "pad_vocab_size_multiple": 1,
                "tie_embeddings": False,
                "lm_head_bias": block_name == "MambaPhi",
            },
            "MixerModel": {
                "name": mixer_name,
                "input": _base_input(),
                "Blocks": [
                    {
                        "name": block_name,
                        "n_layers": 1,
                        "input": (
                            {"norm_epsilon": 1e-5, "mlp_intermediate_size": 32, "mlp_act_fn": "silu"}
                            if block_name in {"LlamaBlock", "Qwen2Block"}
                            else {"resid_dropout": 0.0}
                        ),
                        "Layer": layer,
                    }
                ],
            },
        }
    )


@pytest.mark.parametrize(
    "family",
    [
        ("LlamaModel", "LlamaBlock", _attention_layer("LlamaAttention")),
        ("Qwen2Model", "Qwen2Block", _attention_layer("Qwen2Attention")),
        (
            "LlamaModel",
            "MambaPhi",
            {
                "name": "PhiAttention",
                "input": {
                    "flash_attention": False,
                    "hidden_size": 16,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "partial_rotary_factor": 0.5,
                    "max_position_embeddings": 16,
                    "rope_theta": 10000.0,
                },
            },
        ),
        (
            "LlamaModel",
            "FalconBlock",
            {
                "name": "FalconMambaMixer",
                "input": {"state_size": 4, "conv_kernel": 4, "expand": 1},
            },
        ),
    ],
)
def test_tiny_local_model_family_forward_backward_step_checkpoint_reload(family):
    cfg = _family_config(family)
    model = build_local_model(
        cfg,
        dtype=torch.float32,
        device="cpu",
        attn_implementation="eager",
    )

    input_ids = torch.tensor([[1, 2, 3, 4]])
    outputs = model(input_ids=input_ids, return_hidden_states=True)
    hf_style_outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        output_attentions=True,
    )
    loss = outputs.logits.float().mean()
    loss.backward()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/pytorch_model.bin"
        torch.save(model.state_dict(), path)
        reloaded = build_local_model(
            cfg,
            dtype=torch.float32,
            device="cpu",
            attn_implementation="eager",
        )
        reloaded.load_state_dict(torch.load(path, map_location="cpu"))

    assert outputs.logits.shape == (1, 4, 32)
    assert len(hf_style_outputs.all_hidden_states) == 2
    assert len(hf_style_outputs.all_transfer_matrices) in {0, 1}
