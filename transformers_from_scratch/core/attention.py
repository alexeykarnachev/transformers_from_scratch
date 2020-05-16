from typing import Optional

import torch
from torch import nn

INF = float('inf')


class AttentionError(Exception):
    pass


class MultiHeadAttentionOutput(nn.Module):
    def __init__(self, dim: int, layer_norm_eps: float):
        super().__init__()

        self._linear = nn.Linear(dim, dim)
        self._layer_norm = nn.LayerNorm(dim, eps=layer_norm_eps)

    def forward(
            self,
            attention_output: torch.Tensor,
            attention_input: torch.Tensor
    ) -> torch.Tensor:
        output = self._linear(attention_output)
        output = self._layer_norm(output + attention_input)

        return output


class MultiHeadMaskedAttention(nn.Module):
    def __init__(self, n_heads: int, dim: int, layer_norm_eps:float):
        super().__init__()

        self._n_heads = n_heads
        self._dim = dim
        self._layer_norm_eps = layer_norm_eps

        self._head_dim = self._get_head_dim()
        self._scale = self._head_dim ** -0.5

        self._q_proj = nn.Linear(self._dim, self._dim)
        self._k_proj = nn.Linear(self._dim, self._dim)
        self._v_proj = nn.Linear(self._dim, self._dim)

        self._output = MultiHeadAttentionOutput(
            dim=self._dim,
            layer_norm_eps=self._layer_norm_eps
        )

    def _get_head_dim(self):
        head_dim = self._dim // self._n_heads
        if head_dim * self._n_heads != self._dim:
            raise AttentionError("`dim` must be divisible by `n_heads`.")

        return head_dim

    def forward(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            attention_ignore_mask: Optional[torch.Tensor] = None
    ):
        # Projections (bs, seq_len, dim):
        q = self._q_proj(queries)
        k = self._k_proj(keys)
        v = self._v_proj(values)

        # Multi-head views (bs, seq_len, n_heads, head_dim):
        q = self._get_multi_head_view(q)
        k = self._get_multi_head_view(k)
        v = self._get_multi_head_view(v)

        # Scaled multi-head attention scores (bs, n_heads, seq_len, seq_len):
        scores = q @ k.transpose(-1, -2)
        scores *= self._scale

        # Mask attention scores if needed (bs, n_heads, seq_len, seq_len):
        if attention_ignore_mask is not None:
            scores -= (attention_ignore_mask * INF)

        # Multi-Head attention distributions (bs, n_heads, seq_len, seq_len):
        alphas = nn.functional.softmax(scores, dim=-1)

        # Weighted multi-head values (bs, n_heads, head_dim, seq_len):
        v = alphas @ v

        # Restore original shape (bs, seq_len, dim):
        v = v.transpose(1, 2).contiguous()
        v = v.view(*v.size()[:-2], -1)

        # Output projection (bs, seq_len, dim):
        v = self._output(attention_input=values, attention_output=v)

        return v

    def _get_multi_head_view(self, inp: torch.Tensor) -> torch.Tensor:
        bs, seq_len, dim = inp.size()
        inp_view = inp.view((bs, seq_len, self._n_heads, self._head_dim))
        return inp_view.transpose(1, 2)


class MultiHeadCausalAttention(nn.Module):
    def __init__(
            self,
            n_heads: int,
            dim: int,
            max_seq_len: int,
            layer_norm_eps: float
    ):
        super().__init__()
        self._max_seq_len = max_seq_len
        self._register_attention_ignore_mask()

        self._attention = MultiHeadMaskedAttention(
            n_heads=n_heads,
            dim=dim,
            layer_norm_eps=layer_norm_eps
        )

    def _register_attention_ignore_mask(self) -> None:
        mask = torch.ones((self._max_seq_len, self._max_seq_len))
        mask = torch.tril(mask)
        self.register_buffer('_attention_ignore_mask', mask)

    def forward(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor
    ):
        output = self._attention(
            queries=queries,
            keys=keys,
            values=values,
            attention_ignore_mask=self._attention_ignore_mask
        )

        return output
