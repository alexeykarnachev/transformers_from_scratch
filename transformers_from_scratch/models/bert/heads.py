import torch
from torch import nn

from transformers_from_scratch.core.heads import Head
from transformers_from_scratch.core.structures import EncoderOutput


class BertNextSentencePredictionHead(Head):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self._pooler = nn.Linear(hidden_dim, hidden_dim)
        self._act = nn.functional.tanh
        self._clf = nn.Linear(hidden_dim, 2)

    def _get_head_input(self, encoder_output: EncoderOutput) -> torch.Tensor:
        return encoder_output.hidden_states[-1][:, 0, :]

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
