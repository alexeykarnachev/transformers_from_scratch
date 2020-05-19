import abc
from typing import Sequence, Union, Optional

import torch

from transformers_from_scratch.core.data.encoding import Encoding
from transformers_from_scratch.core.modelling.structures import BackboneInput


class EncodingsCollate:
    def __init__(self, pad_value: int):
        self._pad_value = pad_value

    def __call__(
            self,
            encodings: Sequence[Encoding],
            device: Optional[Union[torch.device, str]] = None
    ) -> BackboneInput:
        model_input = self._collate(
            encodings=encodings,
            device=device
        )

        return model_input

    @abc.abstractmethod
    def _collate(
            self,
            encodings: Sequence[Encoding],
            device: Optional[Union[torch.device, str]]
    ) -> BackboneInput:
        pass
