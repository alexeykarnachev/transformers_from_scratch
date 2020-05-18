from functools import reduce
from typing import Mapping

import torch
from torch import nn

from transformers_from_scratch.core.encoder import Encoder
from transformers_from_scratch.core.heads import Head
from transformers_from_scratch.core.structures import EncoderInput, ModelOutput


class Model(nn.Module):
    def __init__(self, encoder: Encoder, heads: Mapping[str, Head]):
        super().__init__()
        self._encoder = encoder
        self._heads = nn.ModuleDict(heads)

    def forward(
            self,
            encoder_input: EncoderInput,
            head_labels: Mapping[str, torch.Tensor]
    ) -> ModelOutput:
        encoded = self._encoder(encoder_input=encoder_input)

        head_losses = dict()

        for name, head in self._heads.items():
            labels = head_labels[name]
            head_loss = head(encoded, labels=labels)
            head_losses[name] = head_loss

        loss = self._merge_head_losses(head_losses)

        out = ModelOutput(loss=loss, head_losses=head_losses)

        return out

    @staticmethod
    def _merge_head_losses(
            head_losses: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        return reduce(lambda a, b: a + b, head_losses.values())
