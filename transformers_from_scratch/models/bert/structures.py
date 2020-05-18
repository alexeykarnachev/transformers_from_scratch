from dataclasses import dataclass
from typing import Sequence, Optional

import torch

from transformers_from_scratch.core.structures import (
    EncoderInput,
    EncoderOutput
)


@dataclass
class BertEncoderConfig:
    hidden_dim: int
    n_heads: int
    layer_norm_eps: float
    intermediate_dim: int
    layer_norm_eps: float
    n_layers: int
    n_pos: int
    n_types: int
    vocab_size: int
    pad_token_id: int


@dataclass
class BertEncoderOutput(EncoderOutput):
    attentions: Sequence[torch.Tensor]


@dataclass
class BertEncoderInput(EncoderInput):
    token_type_ids: Optional[torch.Tensor]
    token_pos: Optional[torch.Tensor]
