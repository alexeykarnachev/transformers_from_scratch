from dataclasses import dataclass
from typing import Sequence, Mapping

import torch


@dataclass
class BackboneOutput:
    hidden_states: Sequence[torch.Tensor]


@dataclass
class BackboneInput:
    token_ids: torch.Tensor


@dataclass
class ModelOutput:
    loss: torch.Tensor
    head_losses: Mapping[str, torch.Tensor]
