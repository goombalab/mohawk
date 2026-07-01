import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from .discrete_mamba2_ref import materialize_mixer

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
except ImportError:
    RMSNorm = None

from einops import repeat
from functools import partial

from components._repo_path import ensure_repo_root_on_path

ensure_repo_root_on_path()

from utils.init_weights import _init_weights

from utils.config import Config

class Mixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        n_qk_heads=32,
        n_v_heads=32,
        d_conv=4,
        expand=1,
        norm_cls="rms",
        activation="swish",
        bias=False,
        conv_bias=True,
        chunk_size=128,
        use_ref_impl=False,
        layer_idx=None,
        device='cpu',
        dtype=torch.float32,
        initializer: dict = None,
        **kwargs, # Absorb kwarg for general module
    ):
        """
        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args.
        Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.n_qk_heads = n_qk_heads
        self.n_v_heads = n_v_heads
        self.headdim = self.d_inner // self.n_v_heads
        assert self.n_v_heads == self.d_inner // self.headdim
        assert self.d_inner % self.headdim == 0
        assert self.n_v_heads % self.n_qk_heads == 0
        self.norm_cls = norm_cls
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_ref_impl = use_ref_impl
        self.layer_idx = layer_idx
        self.bias = bias
        self.weight_init_cfg = initializer
        self.factory_kwargs = {"device": device, "dtype": dtype}

        # Projections
        self.in_proj = nn.Linear(
            self.d_model,
            2 * self.d_inner + 2 * self.n_qk_heads * self.d_state + self.n_v_heads,
            bias=bias,
            **self.factory_kwargs,
        )
        self.z_bias = (
            nn.Parameter(torch.zeros(self.d_inner, **self.factory_kwargs)) if not bias else 0
        )  # make sure z_bias always exists

        # Convolutional layer
        conv_dim = self.d_inner + 2 * self.n_qk_heads * self.d_state
        self.conv_bias = conv_bias
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **self.factory_kwargs,
        )

        # Activation after conv
        if self.activation in [None, "id", "identity", "linear", "none"]:
            self.act = nn.Identity()
        else:
            self.act = ACT2FN[self.activation]

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.n_v_heads, **self.factory_kwargs))
        self.D._optim = {"weight_decay": 0.0}

        # Norm before out_proj
        if self.norm_cls in ["rms", "rmsnorm"]:
            assert RMSNorm is not None
            self.norm = RMSNorm(self.d_inner, eps=1e-5, **self.factory_kwargs)
        elif self.norm_cls in ["layer", "layernorm"]:
            self.norm = nn.LayerNorm(self.d_inner, eps=1e-5, **self.factory_kwargs)
        elif self.norm_cls in ["none", "identity"]:
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm class {self.norm_cls}")

        # out_proj
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **self.factory_kwargs
        )

        self.rotary_emb = LlamaRotaryEmbedding(config=Config.from_dict(kwargs))

        # Initializations
        self.apply(
            partial(
                _init_weights,
                n_layer=24,
            )
        )
        self.init_z()
        self.init_conv1d()
        self.init_out_proj()

    @property
    def d_output(self):
        return self.d_model

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor

    def forward(self, u, return_mixer_matrix=False, inference_params=None, position_embeddings=None, **kwargs):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        outputs = {}
        # assert state is None
        batch, seqlen, dim = u.shape

        state = None
        if inference_params is not None:
            state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # States are updated inplace
                u = u.squeeze(1) if len(u.shape) == 3 else u
                out, _ = self.step(u, state)
                out = out.unsqueeze(1) if len(u.shape) == 2 else out
                return {"hidden_states": out}

        # Hacky way to initialize state during inference
        chunk_size = self.chunk_size if state is None else seqlen

        # Pad input to nearest multiple of chunklen
        padded_len = (1 + (seqlen - 1) // chunk_size) * chunk_size
        u = F.pad(u, (0, 0, 0, padded_len - seqlen))

        # Project input
        xBCzA_log = self.in_proj(u)
        xBC, z, A_log = torch.split(
            xBCzA_log,
            [
                self.d_inner + 2 * self.n_qk_heads * self.d_state,
                self.d_inner,
                self.n_v_heads,
            ],
            dim=-1,
        )

        if state is not None:
            # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            xBC_t = rearrange(xBC[:, :seqlen, :], "b l d -> b d l")
            state["conv"].copy_(
                F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0))
            )  # Update state (B D W)

        # Convolutional layer
        xBC = self.convolutional_forward(xBC, padded_len)

        x, B, C = torch.split(
            xBC,
            [
                self.d_inner,
                self.n_qk_heads * self.d_state,
                self.n_qk_heads * self.d_state,
            ],
            dim=-1,
        )

        x = rearrange(x, "b l (h n) -> b l h n", h=self.n_v_heads, n=self.headdim)
        B = rearrange(B, "b l (h n) -> b l h n", h=self.n_qk_heads, n=self.d_state)
        C = rearrange(C, "b l (h n) -> b l h n", h=self.n_qk_heads, n=self.d_state)

        # Positional encoding
        # cos, sin = position_embeddings
        position_ids = torch.arange(0, u.shape[1], device=u.device).unsqueeze(0)
        cos, sin = self.rotary_emb(C, position_ids)
        x, B = apply_rotary_pos_emb(q=x, k=B, cos=cos, sin=sin, unsqueeze_dim=2)

        # SSM forward
        result = mamba_chunk_scan_combined(
            x=x / F.softplus(A_log).to(x.dtype).unsqueeze(-1),
            dt=A_log,
            dt_softplus=True,
            A=-torch.ones(self.n_v_heads, device=A_log.device), #todo: should be nheads
            B=B,
            C=C,
            chunk_size=chunk_size,
            # initial_states=(state["ssm"] if state is not None else None), # currently not supported by mamba_ssm.utils.generation
            return_final_states=(state is not None),
        )

        if state is not None:
            y, ssm_state = result
            state["ssm"].copy_(ssm_state)
        else:
            y = result

        Du = torch.einsum("h,blhp->blhp", self.D, x)
        y = rearrange(y + Du, "b l h p -> b l (h p)")

        # Norm and gate
        y = (
            self.norm(y, z)
            if self.norm_cls in ["rms", "rmsnorm"]
            else self.norm(y) * F.silu(z + self.z_bias)
        )

        # Out proj
        out = self.out_proj(y)
        outputs["hidden_states"] = out[:, :seqlen, :]

        if return_mixer_matrix:
            outputs["transfer_matrix"] = materialize_mixer(
                A_log=A_log, B=B, C=C, D=self.D
                )
        return outputs

    def step(self, u, state, **kwargs):
        """
        u: (B D)
        state: dict of states
        Returns: same shape as u
        """

        # Project input
        xBCzA_log = self.in_proj(u)
        xBC, z, A_log = torch.split(
            xBCzA_log,
            [
                self.d_inner + 2 * self.n_qk_heads * self.d_state,
                self.d_inner,
                self.n_v_heads,
            ],
            dim=-1,
        )

        xBC, conv_state = self.convolutional_step(xBC, state["conv"])
        state["conv"].copy_(conv_state) # update state in place

        x, B, C = torch.split(
            xBC,
            [
                self.d_inner,
                self.n_qk_heads * self.d_state,
                self.n_qk_heads * self.d_state,
            ],
            dim=-1,
        )

        x = rearrange(x, "b (h s) -> b h s", h=self.n_v_heads)
        B = rearrange(B, "b (h s) -> b h s", h=self.n_qk_heads)
        C = rearrange(C, "b (h s) -> b h s", h=self.n_qk_heads)

        state["ssm"] = state["ssm"].to(x.dtype)
        zeros = torch.zeros((self.n_v_heads, self.headdim), device=A_log.device).to(dtype=x.dtype)
        ones = torch.ones((self.n_v_heads, self.headdim, self.d_state), device=A_log.device).to(dtype=x.dtype)
        y = selective_state_update(
            x=x/F.softplus(A_log).to(x.dtype).unsqueeze(-1),
            dt=repeat(A_log, "b h -> b h p", p=self.headdim),
            dt_softplus=True,
            A=-ones,
            B=B,
            C=C,
            state=state["ssm"], # will be updated in place
            dt_bias=zeros,
            D=zeros,
        )

        y = y + self.D[:, None] * x
        y = rearrange(y, "b h p -> b (h p)")

        # Norm and gate
        y = (
            self.norm(y, z)
            if self.norm_cls in ["rms", "rmsnorm"]
            else self.norm(y) * F.silu(z + self.z_bias)
        )
        out = self.out_proj(y)

        return out, state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.in_proj.weight.device
        # conv_state:
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_conv,
            self.conv1d.weight.shape[0],
            device=device,
            dtype=conv_dtype,
        ).transpose(1, 2)
        # ssm_state:
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size,
            self.n_v_heads,
            self.headdim,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return {"conv": conv_state, "ssm": ssm_state}

    def _get_states_from_cache(
        self, inference_params, batch_size, initialize_states=False
    ):
        """
        conv_state: (batch, d_conv, conv1d.weight.shape[0])
        ssm_state: (batch, n_qk_heads, headdim, d_state)
        """
        assert self.layer_idx is not None
        # Allocate memory if not exists
        if self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[
                self.layer_idx
            ] = self.allocate_inference_cache(
                batch_size, inference_params.max_seqlen, dtype=torch.float32
            )
        # Get states
        states = inference_params.key_value_memory_dict[self.layer_idx]
        if initialize_states:
            states["conv"].zero_()
            states["ssm"].zero_()
        return states

    def convolutional_forward(self, xBC, padded_len):
        xBC_fast = xBC.transpose(1, 2)
        fast_conv_supported = (
            causal_conv1d_fn is not None
            and self.activation in ["silu", "swish", "identity"]
            and xBC_fast.stride(0) % 8 == 0
            and xBC_fast.stride(2) % 8 == 0
        )
        if not fast_conv_supported:
            xBC = self.act(
                self.conv1d(xBC.transpose(1, 2))[..., :padded_len].transpose(1, 2)
            )
        else:
            xBC = causal_conv1d_fn(
                xBC_fast,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                activation=None if self.activation == "identity" else self.activation,
            ).transpose(1, 2)
        return xBC

    def convolutional_step(self, xBC, conv_state):
        # Convolutional layer
        conv_state = conv_state.to(xBC.dtype)
        if causal_conv1d_update:
            # from causal_conv1d.causal_conv1d_interface import causal_conv1d_update_ref
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation if self.activation != "identity" else None,
            )
        else:
            conv_state.copy_(
                torch.roll(conv_state, shifts=-1, dims=-1)
            )  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.conv_bias:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(xBC.dtype)  # Some activations change dtype

        return xBC, conv_state


    ##################################################################################
    ############################ PARAMETER INITIALIZATION ############################
    ##################################################################################

    def init_z(self):
        init_name = self.weight_init_cfg.get("z", None)

        if init_name == "default":
            return
        elif init_name == "identity":
            weight = torch.zeros((self.d_inner, self.d_model), **self.factory_kwargs)
            bias = torch.ones(self.d_inner, **self.factory_kwargs) * 1.278464542761
        elif init_name == "zeros":
            weight = torch.zeros((self.d_inner, self.d_model), **self.factory_kwargs)
            bias = None
        else:
            raise NotImplementedError(f"Unknown z_init: {init_name}) ")

        # start after xBC to get to z:
        start = self.d_inner + self.n_qk_heads * self.d_state * 2
        end = start + self.d_inner

        # Handle weight
        data = self.in_proj.weight.data
        data[start:end] = weight
        self.in_proj.weight = nn.Parameter(data)

        # Handle bias
        self.z_bias = nn.Parameter(bias)

        # self.z_proj.weight = nn.Parameter(weight)
        # self.z_proj.bias = nn.Parameter(bias)

    def init_out_proj(self):

        init_name = self.weight_init_cfg.get("out", None)

        if init_name == "default":
            return
        elif init_name == "zeros":
            zeros = torch.zeros((self.d_inner, self.d_model), **self.factory_kwargs)
            self.out_proj.weight = nn.Parameter(zeros)
            self.out_proj.bias = None
        elif init_name == "kaiming":
            import math

            p = self.out_proj.weight
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            with torch.no_grad():
                n_residuals_per_layer = 1
                n_layer = 24
                p /= math.sqrt(n_residuals_per_layer * n_layer)
        else:
            raise NotImplementedError(f"Unknown init_out_proj: {init_name}) ")

    def init_conv1d(self):
        init_name = self.weight_init_cfg.get("convolution", None)
        device = self.factory_kwargs["device"]

        if init_name == "default":
            return
        elif init_name == "ones":
            weight = torch.ones(self.conv1d.weight.shape, **self.factory_kwargs)
        elif init_name == "identity":
            weight = torch.zeros(self.conv1d.weight.shape, **self.factory_kwargs)
            weight.data.index_fill_(-1, torch.tensor([-1], device=device), 1.0)
        else:
            raise ValueError(f"Unknown convolution initialization: {init_name}")

        if self.conv1d.bias is None:
            bias = None
        else:
            bias = torch.zeros(self.conv1d.bias.shape, **self.factory_kwargs)

        self.conv1d.weight = nn.Parameter(weight)
        self.conv1d.bias = nn.Parameter(bias)
