import abc

import torch
from torch import nn

from transformers_from_scratch.core.structures import BackboneOutput


class Head(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            backbone_output: BackboneOutput,
            labels: torch.Tensor
    ) -> torch.Tensor:
        head_input = self._get_head_input(backbone_output=backbone_output)
        logits = self._calc_logits(head_input=head_input)
        loss = self._calc_loss(logits=logits, labels=labels)

        return loss

    @abc.abstractmethod
    def _get_head_input(self, backbone_output: BackboneOutput) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def _calc_logits(self, head_input: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def _calc_loss(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        pass


class TokenClassificationHead(Head):
    def __init__(self, hidden_dim: int, n_classes: int):
        super().__init__()
        self._n_classes = n_classes
        self._linear = nn.Linear(hidden_dim, self._n_classes)

    def _get_head_input(self, backbone_output: BackboneOutput) -> torch.Tensor:
        return backbone_output.hidden_states[-1]

    def _calc_logits(self, head_input: torch.Tensor) -> torch.Tensor:
        return self._linear(head_input)

    def _calc_loss(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        loss = nn.functional.cross_entropy(
            input=logits.view(-1, self._n_classes),
            target=labels.view(-1)
        )

        return loss
