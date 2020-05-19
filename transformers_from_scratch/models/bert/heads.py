from typing import Optional

import torch
from torch import nn
from torch.nn import LayerNorm

from transformers_from_scratch.core.heads import Head, TokenClassificationHead
from transformers_from_scratch.core.structures import BackboneOutput


class BertNextSentencePredictionHead(Head):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self._pooler = nn.Linear(hidden_dim, hidden_dim)
        self._act = nn.functional.tanh
        self._clf = nn.Linear(hidden_dim, 2)

    def _get_head_input(self, backbone_output: BackboneOutput) -> torch.Tensor:
        return backbone_output.hidden_states[-1][:, 0, :]

    def _calc_logits(self, head_input: torch.Tensor) -> torch.Tensor:
        pooled = self._pooler(head_input)
        pooled = self._act(pooled)
        logits = self._clf(pooled)

        return logits

    def _calc_loss(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        return nn.functional.cross_entropy(input=logits, target=labels)


class BertLMPredictionHead(TokenClassificationHead):
    def __init__(
            self,
            hidden_dim: int,
            layer_norm_eps: float,
            vocab_size: int,
            token_emb_weights: Optional[torch.nn.Parameter] = None
    ):
        super().__init__(hidden_dim=hidden_dim, n_classes=vocab_size)

        self._linear = nn.Linear(hidden_dim, hidden_dim)
        self._layer_norm = LayerNorm(hidden_dim, eps=layer_norm_eps)
        self._decoder = nn.Linear(hidden_dim, vocab_size)

        if token_emb_weights is not None:
            self._decoder.weight = token_emb_weights

    def _calc_logits(self, head_input: torch.Tensor) -> torch.Tensor:
        logits = self._linear(head_input)
        logits = nn.functional.gelu(logits)
        logits = self._layer_norm(logits)
        logits = self._decoder(logits)

        return logits
