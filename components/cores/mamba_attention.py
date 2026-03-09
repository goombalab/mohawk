import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

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


class DiscreteMamba2(nn.Module):
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

        # xBC
        # self.xBC_proj = nn.Linear(self.d_model, self.d_inner + self.nheads * self.d_state * 2, bias=False, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_model, self.d_inner, bias=False, **factory_kwargs
        )
        self.B_proj = nn.Linear(
            self.d_model, self.nheads * self.d_state, bias=False, **factory_kwargs
        )
        self.C_proj = nn.Linear(
            self.d_model, self.nheads * self.d_state, bias=False, **factory_kwargs
        )

        # A_log
        self.A_log_proj = nn.Linear(
            self.d_model, self.nheads, bias=False, **factory_kwargs
        )

        # D "skip" parameter
        self.D = nn.Parameter(
            torch.ones(self.d_inner if self.D_has_hdim else self.nheads, device=device)
        )
        self.D._optim = {"weight_decay": 0.0}

        # gate:
        self.z_proj = nn.Linear(self.d_model, self.d_inner, bias=True, **factory_kwargs)

        # out_proj
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=True, **factory_kwargs
        )

        # Rotary embeddings (with Phi's configured constants)
        self.rotary_emb = PhiRotaryEmbedding(
            int(0.5 * self.headdim),
            max_position_embeddings=2048,
            base=10000.0,
        )

        # Convolutional layer
        conv_dim = (
            self.d_inner + self.nheads * self.d_state * 2
            if self.parallel_proj
            else self.d_inner
        )
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

        # xBC = self.xBC_proj(u)
        x = self.x_proj(u)
        B = self.B_proj(u)
        C = self.C_proj(u)
        # xBC = torch.cat([x, B, C], dim=-1) if self.parallel_proj else x
        # xBC = self.convolutional_forward(xBC, padded_len)
        # x, B, C = torch.split(xBC, [self.d_inner, self.nheads*self.d_state, self.nheads*self.d_state], dim=-1)
        B, C = self.rotary_pos_emb(x, B, C, device=u.device)
        A_log = self.A_log_proj(u)  # log(A) = 0 ==>> A = 1

        y, T = self.ssm(x, B, C, A_log, return_mixer_matrix)
        z = self.z_proj(u)
        if self.norm_cls in ["rms", "rmsnorm"]:
            y = self.norm(y, z)
        else:
            y = self.norm(y) * F.silu(z)
        out = self.out_proj(y)
        out = out[:, :seqlen, :]

        # store outputs
        outputs["hidden_states"] = out
        if return_mixer_matrix:
            outputs["transfer_matrix"] = T
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

    def convolutional_continue(self, xBC, conv_state):
        assert self.d_conv > 0
        if causal_conv1d_update is not None and self.use_step_conv_kernel:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )
        else:
            # conv_state = torch.roll(conv_state, shifts=-1, dims=-1)  # Update state (B D W)
            conv_state.copy_(
                torch.roll(conv_state, shifts=-1, dims=-1)
            )  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            # (B D)
            xBC = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.conv_bias:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC)  # .to(x.dtype) # Some activations change dtype

        return xBC

    def ssm(self, x, B, C, A_log, return_mixer_matrix=False):
        """
        Arguments:
            x: (batch, seqlen, nheads, headdim)
            B: (batch, seqlen, nheads, dstate)
            C: (batch, seqlen, nheads, dstate)
            A_log: (batch, seqlen, nheads)

        Return:
            y: (batch, seqlen, nheads, headdim)
            T: (batch, nheads, seqlen, seqlen)
        """
        B, C = [rearrange(M, "b l (h n) -> b l h n", h=self.nheads) for M in (B, C)]
        T = None
        batch_size, seq_length = x.shape[:2]
        if return_mixer_matrix:
            # Since the transfer matrix will be equated to the attention matrix,
            # we need to support the form: torch.matmul(attn_weights, value_states)
            T = self.to_transfer_matrix_ssd(A_log=A_log, B=B, C=C, D=self.D)
            T = rearrange(T, "b h z l -> b h l z")
            T = T / math.sqrt(self.headdim)
            attention_mask = _prepare_4d_causal_attention_mask(
                None, (batch_size, seq_length), x, 0
            )
            T = T + attention_mask
            T = nn.functional.softmax(T, dim=-1, dtype=torch.float32).to(
                x.dtype
            )  # added for testing
            X = rearrange(x, "b l (h p) -> b h l p", h=self.nheads, p=self.headdim)
            y = torch.matmul(T, X)
            y = rearrange(y, "b h l p -> b l (h p)")
        elif self.use_ref_impl:
            X = rearrange(x, "b l (h p) -> b l h p", h=self.nheads, p=self.headdim)
            y, final_state = self.ssd_minimal_discrete(
                X=X, A_log=A_log, B=B, C=C, block_len=self.chunk_size
            )
            Du = torch.einsum("h,blhp->blhp", self.D, X)
            y = rearrange(y + Du, "b l h p -> b l (h p)")
        else:
            X = rearrange(x, "b l (h p) -> b l h p", h=self.nheads, p=self.headdim)
            y = mamba_chunk_scan_fused(
                x=X / A_log.unsqueeze(-1),  # (batch, seqlen, nheads, headdim)
                dt=rearrange(A_log, "b (c l) h -> b h c l", l=self.chunk_size),
                A=torch.ones(self.nheads, device=A_log.device),
                B=B,  # (batch, seqlen, nheads, dstate)
                C=C,
                D=None,
                z=None,
            )
            Du = torch.einsum("h,blhp->blhp", self.D, X)
            y = rearrange(y + Du, "b l h p -> b l (h p)")

        return y, T

    def default_state(
        self, *batch_shape, device=None, dtype=None, inplace_state=None, **kwargs
    ):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        device = self.A_log.device if device is None else device
        if self.d_conv > 0:
            conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
            conv_state = torch.zeros(
                (
                    *batch_shape,
                    self.d_inner + self.nheads * self.d_state * 2,
                    self.d_conv,
                ),
                device=device,
                dtype=conv_dtype,
            )
        else:
            conv_state = None
        # layer_state = self.layer.default_state(*batch_shape)
        ssm_dtype = self.A_log.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            (*batch_shape, self.nheads, self.headdim, self.d_state),
            device=device,
            dtype=ssm_dtype,
        )
        if inplace_state:
            inplace_conv_state, inplace_ssm_state = inplace_state
            inplace_conv_state.copy_(conv_state)
            inplace_ssm_state.copy_(ssm_state)
            conv_state, ssm_state = inplace_conv_state, inplace_ssm_state
        return (conv_state, ssm_state)

    @property
    def d_output(self):
        return self.d_model

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor

    def allocate_inference_cache(self, batch_size, dtype=None, **kwargs):
        return self.default_state(batch_size, device=None, dtype=dtype)

    def _get_states_from_cache(
        self, inference_params, batch_size, initialize_states=False
    ):
        assert self.layer_idx is not None
        # TODO: I don't like this design pattern, the states should be allocated up front
        # and handled locally instead of having a global state
        if self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[
                self.layer_idx
            ] = self.allocate_inference_cache(batch_size)
        conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
        # TODO: What if batch size changes between generation, and we reuse the same states?
        if initialize_states:
            conv_state.zero_()
            ssm_state.zero_()
        return conv_state, ssm_state

    @staticmethod
    def segsum(x):
        """More stable segment sum calculation."""
        # [1, 2, 3]
        T = x.size(-1)
        x = repeat(x, "... d -> ... d e", e=T)
        # [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        # [[0, 0, 0], [2, 0, 0], [3, 3, 0]]
        x_segsum = torch.cumsum(x, dim=-2)
        # [[0, 0, 0], [2, 0, 0], [5, 3, 0]]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    @staticmethod
    def to_transfer_matrix_ssd(A_log, B, C, D):
        """
        Arguments:
            A_log: (batch, length, n_heads)
            B: (batch, length, n_heads, d_state)
            C: (batch, length, n_heads, d_state)
        Return:
            T: (batch, n_heads, length, length)
        """
        batch_size, length, n_heads, d_state = B.shape
        assert A_log.shape == (batch_size, length, n_heads)
        assert B.shape == C.shape == (batch_size, length, n_heads, d_state)
        # Compute:
        A_log = rearrange(A_log, "b l h -> b h l")
        powers = torch.exp(DiscreteMamba2.segsum(A_log))
        T = torch.einsum("blhn,bshn,bhls->bhsl", C, B, powers)
        # Add D:
        if D is not None:
            T[:, :, torch.arange(length), torch.arange(length)] += D.view(1, n_heads, 1)
        return T

    @staticmethod
    def ssd_minimal_discrete(X, A_log, B, C, block_len, initial_states=None):
        """
        Arguments:
            X: (batch, length, n_heads, d_head)
            A_log: (batch, length, n_heads)
            B: (batch, length, n_heads, d_state)
            C: (batch, length, n_heads, d_state)
        Return:
            Y: (batch, length, n_heads, d_head)
            final_state: (batch, n_heads, d_head, d_state)
        """
        assert X.dtype == A_log.dtype == B.dtype == C.dtype
        assert X.shape[1] % block_len == 0
        batch_size, length, n_heads, d_head = X.shape
        d_state = B.shape[-1]
        assert A_log.shape == (batch_size, length, n_heads)
        assert B.shape == C.shape == (batch_size, length, n_heads, d_state)

        # Rearrange into blocks/chunks
        X, A_log, B, C = [
            rearrange(x, "b (c l) ... -> b c l ...", l=block_len)
            for x in (X, A_log, B, C)
        ]

        A_log = rearrange(A_log, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_log, dim=-1)

        # 1. Compute the output for each intra-chunk (diagonal blocks)
        length = torch.exp(DiscreteMamba2.segsum(A_log))
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, length, X)

        # 2. Compute the state for each intra-chunk
        # (right term of low-rank factorization of off-diagonal blocks; B terms)
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

        # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
        # (middle term of factorization of off-diag blocks; A terms)
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        decay_chunk = torch.exp(
            DiscreteMamba2.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))
        )
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states, final_state = new_states[:, :-1], new_states[:, -1]

        # 4. Compute state -> output conversion per chunk
        # (left term of low-rank factorization of off-diagonal blocks; C terms)
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

        # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        return Y, final_state

    def rotary_pos_emb(self, X, B, C, device):
        """
        Query = C
        Key = B
        Value = X
        """
        bsz, q_len, _ = C.shape

        query_states = C
        key_states = B
        value_states = X

        query_states = query_states.view(
            bsz, q_len, self.nheads, self.headdim
        ).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.nheads, self.headdim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.nheads, self.headdim
        ).transpose(1, 2)

        seq_len = kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : self.rotary_emb.dim],
            query_states[..., self.rotary_emb.dim :],
        )
        key_rot, key_pass = (
            key_states[..., : self.rotary_emb.dim],
            key_states[..., self.rotary_emb.dim :],
        )

        position_ids = torch.arange(0, seq_len + 0, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)
        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        query_rot, key_rot = apply_rotary_pos_emb(
            query_rot, key_rot, cos, sin, position_ids
        )

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        query_states = query_states.transpose(1, 2).reshape(bsz, q_len, self.d_inner)
        key_states = key_states.transpose(1, 2).reshape(bsz, q_len, self.d_inner)

        C = query_states
        B = key_states
        return B, C
