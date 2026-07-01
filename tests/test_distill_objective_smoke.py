from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.slow]

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from distill.distill_steps import dpo, hstates, matrices, sequential_hstates, supervised, supervised_instruct
from utils.build_model import build_local_model
from utils.config import Config


def _tiny_llama_components():
    return Config.from_dict(
        {
            "name": "LayeredMambaLM",
            "input": {
                "vocab_size": 32,
                "pad_vocab_size_multiple": 1,
                "tie_embeddings": False,
                "lm_head_bias": False,
            },
            "MixerModel": {
                "name": "LlamaModel",
                "input": {
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
                },
                "Blocks": [
                    {
                        "name": "LlamaBlock",
                        "n_layers": 1,
                        "input": {
                            "norm_epsilon": 1e-5,
                            "mlp_intermediate_size": 32,
                            "mlp_act_fn": "silu",
                        },
                        "Layer": {
                            "name": "LlamaAttention",
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
                        },
                    }
                ],
            },
        }
    )


class TinyWrapper:
    def __init__(self):
        self.module = build_local_model(
            _tiny_llama_components(),
            dtype=torch.float32,
            device="cpu",
            attn_implementation="eager",
        )
        self.model = self
        self.backward_calls = 0

    def __call__(self, *args, **kwargs):
        if "layer_idx" in kwargs:
            return self.module(*args, **kwargs)

        output = self.module(
            input_ids=kwargs.get("input_ids", args[0] if args else None),
            position_ids=kwargs.get("position_ids"),
            return_hidden_states=kwargs.get(
                "return_hidden_states", kwargs.get("output_hidden_states", False)
            ),
            return_mixer_matrix=kwargs.get(
                "return_mixer_matrix", kwargs.get("output_attentions", False)
            ),
        )
        return SimpleNamespace(
            logits=output.logits,
            hidden_states=output.all_hidden_states,
            attentions=output.all_transfer_matrices,
            position_embeddings=output.position_embeddings,
        )

    def backward(self, loss):
        self.backward_calls += 1
        loss.backward()

    def generate(self, input_ids, max_length, **kwargs):
        pad = torch.zeros(
            input_ids.shape[0],
            max_length - input_ids.shape[1],
            dtype=input_ids.dtype,
        )
        return torch.cat([input_ids.cpu(), pad], dim=1)


@pytest.mark.parametrize(
    ("step_module", "batch"),
    [
        (supervised, {"input_ids": torch.tensor([[1, 2, 3, 4]])}),
        (hstates, {"input_ids": torch.tensor([[1, 2, 3, 4]])}),
        (sequential_hstates, {"input_ids": torch.tensor([[1, 2, 3, 4]])}),
        (matrices, {"input_ids": torch.tensor([[1, 2, 3, 4]])}),
        (
            dpo,
            {
                "input_ids": torch.tensor([[1, 2]]),
                "chosen_ids": torch.tensor([[3, 4]]),
                "rejected_ids": torch.tensor([[4, 3]]),
            },
        ),
        (
            supervised_instruct,
            {
                "input_ids": torch.tensor([[1, 2]]),
                "response_ids": torch.tensor([[3, 4]]),
            },
        ),
    ],
)
def test_distill_objective_with_tiny_real_models_backward_optimizer_scheduler(step_module, batch):
    student = TinyWrapper()
    teacher = TinyWrapper()
    cfg = Config.from_dict({"ComponentsConfig": {"input": {"vocab_size": 32}}})
    optimizer = torch.optim.AdamW(student.module.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    scores = step_module.distill_step(
        batch,
        student,
        teacher,
        cfg,
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=0),
    )
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    assert scores
    assert all(isinstance(score, float) for score in scores)
    assert student.backward_calls == 1
