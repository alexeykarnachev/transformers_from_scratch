from functools import reduce
from typing import Mapping

import torch
from torch import nn

from transformers_from_scratch.core.modelling.backbone import Backbone
from transformers_from_scratch.core.modelling.heads import Head
from transformers_from_scratch.core.modelling.structures import BackboneInput, ModelOutput


class Model(nn.Module):
    def __init__(self, backbone: Backbone, heads: Mapping[str, Head]):
        super().__init__()
        self._backbone = backbone
        self._heads = nn.ModuleDict(heads)

    def forward(
            self,
            backbone_input: BackboneInput,
            head_labels: Mapping[str, torch.Tensor]
    ) -> ModelOutput:
        encoded = self._backbone(backbone_input=backbone_input)

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
