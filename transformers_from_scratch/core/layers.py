import abc
import math
from typing import Callable

import torch
from torch import nn

from transformers_from_scratch.core.functions import apply_attention_distr


class AddAndNorm(nn.Module):
    def __init__(self, dim: int, layer_norm_eps: float):
        super().__init__()

        self._layer_norm = nn.LayerNorm(dim, eps=layer_norm_eps)

    def forward(self, inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        # inp, out: (bs, seq_len, dim)
        added = inp + out
        normed = self._layer_norm(added)

        return normed


class MultiHeadProjector(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self._dim = dim
        self._n_heads = n_heads
        self._head_dim = self._get_head_dim()

        self._projector = nn.Linear(dim, dim)

    def _get_head_dim(self):
        head_dim = self._dim // self._n_heads
        if head_dim * self._n_heads != self._dim:
            raise ValueError("`dim` must be divisible by `n_heads`.")

        return head_dim

    def forward(self, inp: torch.Tensor):
        out = self._projector(inp)

        # (bs, seq_len, n_heads, head_dim)
        new_shape = inp.size()[:2] + (self._n_heads, self._head_dim)
        out = out.view(new_shape)

        # (bs, n_heads, seq_len, head_dim)
        return out.transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, act_fn: Callable):
        super().__init__()

        self._act_fn = act_fn

        self._ff_inp = nn.Linear(dim, intermediate_dim)
        self._ff_out = nn.Linear(intermediate_dim, dim)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        out = self._ff_inp(inp)
        out = self._act_fn(out)
        out = self._ff_out(out)

        return out


class Attention(nn.Module, abc.ABC):
    def __init__(self, dim: int):
        super().__init__()

        self._out_proj = nn.Linear(dim, dim)

    def forward(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor
    ) -> torch.Tensor:
        # queries, keys: (bs, seq_len, n_heads, head_dim)
        # Scaled multi-head attention scores (bs, n_heads, seq_len, seq_len):
        scores = queries @ keys.transpose(-1, -2)
        scale = 1 / math.sqrt(queries.size()[-1])
        scores *= scale

        # Multi-Head attention distributions (bs, n_heads, seq_len, seq_len):
        att_distr = self._get_attention_distribution(scores=scores)

        # Attend (bs, seq_len, dim):
        out = apply_attention_distr(values=values, distr=att_distr)

        # Output projection (bs, seq_len, dim):
        out = self._out_proj(out)

        return out

    @abc.abstractmethod
    def _get_attention_distribution(self, scores: torch.Tensor) -> torch.Tensor:
        pass


class FullAttention(Attention):
    def __init__(self, dim: int):
        super().__init__(dim=dim)

    def _get_attention_distribution(self, scores: torch.Tensor) -> torch.Tensor:
        # scores, distr: (bs, n_heads, seq_len, seq_len)
        distr = nn.functional.softmax(scores, dim=-1)
        return distr
