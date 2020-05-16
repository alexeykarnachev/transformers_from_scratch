import torch
from torch import nn


class AttentionError(Exception):
    pass


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, dim: int):
        super().__init__()

        self._n_heads = n_heads
        self._dim = dim

        self._head_dim = self._get_head_dim()
        self._scale = self._head_dim ** -0.5

        self._q_proj = nn.Linear(self._dim, self._dim)
        self._k_proj = nn.Linear(self._dim, self._dim)
        self._v_proj = nn.Linear(self._dim, self._dim)

    def _get_head_dim(self):
        head_dim = self._dim // self._n_heads
        if head_dim * self._n_heads != self._dim:
            raise AttentionError("`dim` must be divisible by `n_heads`.")

        return head_dim

    def forward(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor
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

        # Multi-Head attention distributions (bs, n_heads, seq_len, seq_len):
        alphas = nn.functional.softmax(scores, dim=-1)

        # Weighted multi-head values (bs, n_heads, head_dim, seq_len):
        v = alphas @ v

        # Restore original shape:
        v = v.transpose(1, 2).contiguous()
        v = v.view(*v.size()[:-2], -1)

        return v

    def _get_multi_head_view(self, inp: torch.Tensor) -> torch.Tensor:
        bs, seq_len, dim = inp.size()
        inp_view = inp.view((bs, seq_len, self._n_heads, self._head_dim))
        return inp_view.transpose(1, 2)
