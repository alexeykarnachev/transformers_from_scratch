from typing import Optional

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


class BertEmbeddings(nn.Module):
    def __init__(
            self,
            n_pos: int,
            n_types: int,
            vocab_size: int,
            dim: int,
            pad_token_id: int,
            layer_norm_eps: float
    ):
        super().__init__()

        self._token_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            padding_idx=pad_token_id
        )
        self._pos_emb = nn.Embedding(
            num_embeddings=n_pos,
            embedding_dim=dim
        )
        self._type_emb = nn.Embedding(
            num_embeddings=n_types,
            embedding_dim=dim
        )

        self._layer_norm = nn.LayerNorm(dim, eps=layer_norm_eps)

    def forward(
            self,
            token_ids: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            token_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        device = token_ids.device
        inp_shape = token_ids.size()
        seq_len = inp_shape[1]

        if token_pos is None:
            token_pos = torch.arange(seq_len, dtype=torch.long, device=device)
            token_pos = token_pos.unsqueeze(0).expand(inp_shape)

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                inp_shape, dtype=torch.long, device=device)

        token_emb = self._token_emb(token_ids)
        type_emb = self._type_emb(token_type_ids)
        pos_emb = self._pos_emb(token_pos)

        emb = token_emb + type_emb + pos_emb
        out = self._layer_norm(emb)

        return out
