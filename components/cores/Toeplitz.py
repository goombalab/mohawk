import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from Mamba.nn.activation import Activation

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from Mamba.ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm = None


class Mixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        norm_cls="rms",
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=128,
        device=None,
        dtype=None,
        nheads=32,  # nheads must be large, otherwise MambaChunkScanFusedFn.backward will crash during compilation
        **kwargs,
    ):
        """
        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args.
        Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.nheads = nheads
        self.headdim = self.d_inner // self.nheads
        assert self.nheads == self.d_inner // self.headdim
        assert self.d_inner % self.headdim == 0
        self.norm_cls = norm_cls
        self.activation = activation
        self.chunk_size = chunk_size

        self.z_proj = nn.Linear(self.d_model, self.d_inner, bias=True, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_model, self.d_inner, bias=False, **factory_kwargs
        )
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

        if self.norm_cls in ["rms", "rmsnorm"]:
            assert RMSNorm is not None
            self.norm = RMSNorm(self.d_inner, eps=1e-5, **factory_kwargs)
        elif self.norm_cls in ["layer", "layernorm"]:
            self.norm = nn.LayerNorm(self.d_inner, eps=1e-5, **factory_kwargs)
        elif self.norm_cls in ["none", "identity"]:
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm class {self.norm_cls}")

        self.toeplitz_matrix = torch.nn.Conv2d(
            in_channels=self.nheads,
            out_channels=self.nheads,
            kernel_size=(1, 2048),
            groups=self.nheads,
            bias=False,
        )

        self.act = Activation(self.activation)
        conv_dim = self.d_inner  # + self.nheads * self.d_state * 2
        self.conv_bias = conv_bias
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )

    def forward(self, u, **kwargs):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        outputs = {}
        # assert state is None
        batch, seqlen, dim = u.shape

        # Pad input to nearest multiple of chunklen
        padded_len = (1 + (seqlen - 1) // self.chunk_size) * self.chunk_size
        u = F.pad(u, (0, 0, 0, padded_len - seqlen))

        x = self.x_proj(u)
        x = self.convolutional_forward(x, padded_len)
        y = self.apply_toeplitz(x)

        z = self.z_proj(u)
        if self.norm_cls in ["rms", "rmsnorm"]:
            y = self.norm(y, z)
        else:
            y = self.norm(y) * F.silu(z)
        out = self.out_proj(y)
        out = out[:, :seqlen, :]

        # store outputs
        outputs["hidden_states"] = out
        return outputs

    def apply_toeplitz(self, x):
        """
        x: (B, H, T, P)
        A: (B, H)
        B: (B, H)
        """
        X = rearrange(x, "b l (h p) -> b h p l", h=self.nheads, p=self.headdim)
        X = torch.nn.functional.pad(X, (2047, 0, 0, 0))

        y = self.toeplitz_matrix(X)
        y = rearrange(y, "b h p l -> b l (h p)", h=self.nheads, p=self.headdim)

        return y

    def convolutional_forward(self, xBC, padded_len):
        if causal_conv1d_fn is None or self.activation not in [
            "silu",
            "swish",
            "identity",
        ]:
            xBC = self.act(
                self.conv1d(xBC.transpose(1, 2))[..., :padded_len].transpose(1, 2)
            )
        else:
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2),
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                activation=None if self.activation == "identity" else self.activation,
            ).transpose(1, 2)
        return xBC
