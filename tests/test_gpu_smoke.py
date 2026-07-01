import pytest

pytestmark = pytest.mark.cuda

torch = pytest.importorskip("torch")


def test_cuda_device_helper_reports_cuda_when_gpu_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA smoke tests require an NVIDIA GPU")

    from utils.distributed import get_device

    assert get_device().type == "cuda"
