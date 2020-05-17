from typing import Callable

import torch
from torch import nn

from transformers_from_scratch.core.layers import (
    MultiHeadProjector,
    FullAttention,
    AddAndNorm,
    FeedForward
)


class BertLayer(nn.Module):
    def __init__(
            self,
            dim: int,
            n_heads: int,
            layer_norm_eps: float,
            intermediate_dim: int
    ):
        super().__init__()

        self._q_mh_proj = MultiHeadProjector(dim=dim, n_heads=n_heads)
        self._k_mh_proj = MultiHeadProjector(dim=dim, n_heads=n_heads)
        self._v_mh_proj = MultiHeadProjector(dim=dim, n_heads=n_heads)

        self._qk_attn = FullAttention(dim=dim)
        self._add_norm_attn = AddAndNorm(dim=dim, layer_norm_eps=layer_norm_eps)

        self._ff = FeedForward(
            dim=dim,
            intermediate_dim=intermediate_dim,
            act_fn=nn.functional.gelu
        )
        self._add_norm_ff = AddAndNorm(dim=dim, layer_norm_eps=layer_norm_eps)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        q_mh = self._q_mh_proj(inp)
        k_mh = self._k_mh_proj(inp)
        v_mh = self._v_mh_proj(inp)

        out = self._qk_attn(queries=q_mh, keys=k_mh, values=v_mh)
        out = self._add_norm_attn(inp=inp, out=out)

        intermediate = self._ff(out)
        out = self._add_norm_ff(inp=out, out=intermediate)

        return out
