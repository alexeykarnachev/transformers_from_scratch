import abc

from torch import nn

from transformers_from_scratch.core.modelling.structures import (
    BackboneOutput,
    BackboneInput
)


class Backbone(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, encoder_input: BackboneInput) -> BackboneOutput:
        pass
