import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

from Mamba.nn.activation import Activation
from external_models.modeling_phi import (
    PhiRotaryEmbedding,
    _prepare_4d_causal_attention_mask,
    apply_rotary_pos_emb,
)

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from Mamba.ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm = None

from Mamba.ops.triton.flashmamba import mamba_chunk_scan_fused


class Mixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        parallel_proj=True,
        D_has_hdim=False,
        norm_cls="rms",
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=128,
        use_ref_impl=False,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        transposed=False,
        use_step_conv_kernel=True,
        nheads=32,  # nheads must be large, otherwise MambaChunkScanFusedFn.backward will crash during compilation
        n_layer=None,  # Absorb kwarg for general module
        # A_tied=True,
        # use_fast_path=False,
        # dropout=0.0,  # Just to absorb the kwarg
        # headdim=64,
        # conv_init=None,
        # rmsnorm=True,
        **kwargs,
    ):
        """
        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args.
        Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """
        assert not transposed
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
        self.parallel_proj = parallel_proj
        self.D_has_hdim = D_has_hdim
        self.norm_cls = norm_cls
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_ref_impl = use_ref_impl
        self.layer_idx = layer_idx
        self.use_step_conv_kernel = use_step_conv_kernel

        assert self.parallel_proj

        self.x_proj = nn.Linear(
            self.d_model, self.d_inner, bias=False, **factory_kwargs
        )

        # gate:
        self.z_proj = nn.Linear(self.d_model, self.d_inner, bias=True, **factory_kwargs)

        # out_proj
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=True, **factory_kwargs
        )

        self.toeplitz = MultiHeadConv1D(
            self.d_model,
            self.d_state,
            self.d_conv,
            self.expand,
            self.nheads,
            self.headdim,
        )

        # Conv
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

        # Activation after conv
        self.act = Activation(self.activation)

        # Norm before out_proj
        if self.norm_cls in ["rms", "rmsnorm"]:
            assert RMSNorm is not None
            self.norm = RMSNorm(self.d_inner, eps=1e-5, **factory_kwargs)
        elif self.norm_cls in ["layer", "layernorm"]:
            self.norm = nn.LayerNorm(self.d_inner, eps=1e-5, **factory_kwargs)
        elif self.norm_cls in ["none", "identity"]:
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm class {self.norm_cls}")

    def forward(
        self, u, state=None, return_mixer_matrix=False, inference_params=None, **kwargs
    ):
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
        y = self.toeplitz(x)
        z = self.z_proj(u)

        if self.norm_cls in ["rms", "rmsnorm"]:
            y = self.norm(y, z)
        else:
            y = self.norm(y) * F.silu(z)
        out = self.out_proj(y)
        outputs["hidden_states"] = out[:, :seqlen, :]
        return outputs

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


class MultiHeadConv1D(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, nheads, headdim):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.nheads = nheads
        self.headdim = headdim

        # Convolutional layer
        # Size (B, H, T, P)
        self.filter_proj = nn.Linear(d_model, self.nheads, bias=False)
        # size (B, T, H)

    # @torch.compile
    def forward(self, x):
        """
        This implmentation uses fft to compute the convolution.

        To calculate the convolution filter:
        1. Broadcast the filter (B, T, H) to (B, H, T, P)
        2. Move the filter to the frequency domain
        3. Multiply the filter with the input in the frequency domain
        4. Move the result back to the time domain

        Args:
        x: (B, T, D)

        where:
        B = batch size
        H = number of heads
        T = sequence length
        P = head dimension
        D = model dimension

        Returns:
        # y: (B, H, T, P)
        """
        batch, seqlen, dim = x.shape
        dtype = x.dtype
        filter = self.filter_proj(x)
        filter = rearrange(filter, "b t h -> b h t", h=self.nheads).contiguous()
        x = rearrange(x, "b t (h p) -> b h p t", h=self.nheads).contiguous()

        # Move to frequency domain
        filter = F.pad(filter, (seqlen - 1, 0))
        filter = torch.fft.rfft(filter.float()).to(dtype)
        x = F.pad(x, (seqlen - 1, 0))
        x = torch.fft.rfft(x.float()).to(dtype)

        # Multiply
        y = einsum(filter, x, "b h t, b h p t -> b h p t").float()

        # Move back to time domain
        y = torch.fft.irfft(y).to(dtype)[..., :seqlen]
        y = rearrange(y, "b h p t -> b t (h p)", h=self.nheads, p=self.headdim)

        return y
