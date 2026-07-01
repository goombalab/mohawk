from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from utils.config import Config
from distill.distill_steps import dpo, hstates, matrices, sequential_hstates, supervised, supervised_instruct


class FakeModule:
    def __init__(self, n_layers):
        self.forward_fn = "_forward"
        self.backbone = SimpleNamespace(layers=[object() for _ in range(n_layers)])


class FakeWrapper:
    def __init__(self, vocab_size=5, n_layers=2, teacher=False):
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.teacher = teacher
        self.module = FakeModule(n_layers)
        self.model = self
        self.backward_calls = []
        self.param = torch.tensor(0.25, requires_grad=not teacher)

    def backward(self, loss):
        self.backward_calls.append(loss)

    def __call__(self, *args, **kwargs):
        if "layer_idx" in kwargs:
            hidden_states = kwargs["hidden_states"]
            seq_len = hidden_states.shape[1]
            result = {"hidden_states": hidden_states + self.param}
            if kwargs.get("return_mixer_matrix"):
                result["transfer_matrix"] = torch.zeros(
                    hidden_states.shape[0],
                    1,
                    seq_len,
                    seq_len,
                    requires_grad=not self.teacher,
                ) + self.param
            return result

        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            input_ids = args[0]
        batch, seq_len = input_ids.shape
        logits = torch.zeros(
            batch,
            seq_len,
            self.vocab_size,
            requires_grad=not self.teacher,
        ) + self.param
        hidden_states = tuple(
            torch.full((batch, seq_len, 3), float(idx)) for idx in range(self.n_layers + 1)
        )
        attentions = tuple(
            torch.zeros(batch, 1, seq_len, seq_len) for _ in range(self.n_layers)
        )
        return SimpleNamespace(logits=logits, hidden_states=hidden_states, attentions=attentions)

    def generate(self, input_ids, max_length, **kwargs):
        pad = torch.zeros(input_ids.shape[0], max_length - input_ids.shape[1], dtype=input_ids.dtype)
        return torch.cat([input_ids.cpu(), pad], dim=1)


def cfg(vocab_size=5):
    return Config.from_dict({"ComponentsConfig": {"input": {"vocab_size": vocab_size}}})


def test_supervised_returns_numeric_score_and_backprops_once():
    student = FakeWrapper()
    teacher = FakeWrapper(teacher=True)
    scores = supervised.distill_step({"input_ids": torch.tensor([[1, 2, 3]])}, student, teacher, cfg())

    assert len(scores) == 1
    assert isinstance(scores[0], float)
    assert len(student.backward_calls) == 1


def test_hstates_returns_layer_scores_and_backprops_once():
    student = FakeWrapper()
    teacher = FakeWrapper(teacher=True)
    scores = hstates.distill_step({"input_ids": torch.tensor([[1, 2, 3]])}, student, teacher, cfg())

    assert len(scores) == 2
    assert all(isinstance(score, float) for score in scores)
    assert len(student.backward_calls) == 1


def test_sequential_hstates_returns_layer_scores_and_backprops_once():
    student = FakeWrapper()
    teacher = FakeWrapper(teacher=True)
    scores = sequential_hstates.distill_step(
        {"input_ids": torch.tensor([[1, 2, 3]])}, student, teacher, cfg()
    )

    assert len(scores) == 2
    assert len(student.backward_calls) == 1


def test_matrices_returns_layer_scores_and_backprops_once():
    student = FakeWrapper()
    teacher = FakeWrapper(teacher=True)
    scores = matrices.distill_step({"input_ids": torch.tensor([[1, 2, 3]])}, student, teacher, cfg())

    assert len(scores) == 2
    assert len(student.backward_calls) == 1


def test_dpo_returns_numeric_score_and_validates_keys():
    student = FakeWrapper()
    teacher = FakeWrapper(teacher=True)
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "chosen_ids": torch.tensor([[3, 4]]),
        "rejected_ids": torch.tensor([[4, 3]]),
    }

    scores = dpo.distill_step(batch, student, teacher, cfg())

    assert len(scores) == 1
    assert isinstance(scores[0], float)
    assert len(student.backward_calls) == 1

    with pytest.raises(ValueError, match="Could not find chosen response"):
        dpo.distill_step({"input_ids": torch.tensor([[1]])}, student, teacher, cfg())


def test_supervised_instruct_returns_numeric_score_and_backprops_once():
    student = FakeWrapper()
    teacher = FakeWrapper(teacher=True)
    scores = supervised_instruct.distill_step(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "response_ids": torch.tensor([[3, 4]]),
        },
        student,
        teacher,
        cfg(),
        tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=0),
    )

    assert len(scores) == 1
    assert isinstance(scores[0], float)
    assert len(student.backward_calls) == 1
