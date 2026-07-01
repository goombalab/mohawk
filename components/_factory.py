from collections.abc import Mapping
from typing import Any, TypeVar

from torch import nn


ModuleT = TypeVar("ModuleT", bound=nn.Module)


def apply_module_factory_kwargs(
    module: ModuleT, factory_kwargs: Mapping[str, Any]
) -> ModuleT:
    move_kwargs = {
        key: value for key, value in factory_kwargs.items() if value is not None
    }
    if not move_kwargs:
        return module

    tensors = list(module.parameters(recurse=True)) + list(
        module.buffers(recurse=True)
    )
    if any(tensor.device.type == "meta" for tensor in tensors):
        dtype = move_kwargs.get("dtype")
        return module.to(dtype=dtype) if dtype is not None else module

    return module.to(**move_kwargs)
