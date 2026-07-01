import pytest


@pytest.fixture
def torch_runtime():
    torch = pytest.importorskip("torch")
    from components._factory import apply_module_factory_kwargs

    return torch, apply_module_factory_kwargs


def test_apply_module_factory_kwargs_ignores_none_values(torch_runtime):
    torch, apply_module_factory_kwargs = torch_runtime
    module = torch.nn.Linear(2, 2)

    result = apply_module_factory_kwargs(module, {"device": None, "dtype": None})

    assert result is module
    assert module.weight.dtype == torch.float32


def test_apply_module_factory_kwargs_moves_parameters_and_buffers(torch_runtime):
    torch, apply_module_factory_kwargs = torch_runtime

    class ModuleWithBuffer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.register_buffer("scale", torch.ones(2))

    module = ModuleWithBuffer()

    result = apply_module_factory_kwargs(
        module, {"device": torch.device("cpu"), "dtype": torch.float64}
    )

    assert result is module
    assert module.weight.device.type == "cpu"
    assert module.scale.device.type == "cpu"
    assert module.weight.dtype == torch.float64
    assert module.scale.dtype == torch.float64


def test_apply_module_factory_kwargs_does_not_materialize_meta_tensors(
    torch_runtime,
):
    torch, apply_module_factory_kwargs = torch_runtime
    module = torch.nn.Linear(2, 2, device="meta")

    result = apply_module_factory_kwargs(
        module, {"device": torch.device("cpu"), "dtype": torch.float64}
    )

    assert result is module
    assert module.weight.device.type == "meta"
    assert module.weight.dtype == torch.float64
