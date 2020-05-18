import abc

from torch import nn

from transformers_from_scratch.core.structures import (
    EncoderOutput,
    EncoderInput
)


class Encoder(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, encoder_input: EncoderInput) -> EncoderOutput:
        pass
