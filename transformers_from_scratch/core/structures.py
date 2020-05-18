from dataclasses import dataclass
from typing import Sequence, Mapping

import torch


@dataclass
class EncoderOutput:
    hidden_states: Sequence[torch.Tensor]


@dataclass
class EncoderInput:
    token_ids: torch.Tensor


@dataclass
class ModelOutput:
    loss: torch.Tensor
    head_losses: Mapping[str, torch.Tensor]
