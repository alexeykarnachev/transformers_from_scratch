from collections import namedtuple

import pytest
import torch
from transformers.modeling_bert import BertAttention

from transformers_from_scratch.core.attention import MultiHeadMaskedAttention
from transformers_from_scratch.core.utils import seed_everything

HF_CONFIG = namedtuple(
    typename='Config',
    field_names=(
        'hidden_size',
        'num_attention_heads',
        'attention_probs_dropout_prob',
        'output_attentions',
        'layer_norm_eps',
        'hidden_dropout_prob'
    )
)


@pytest.mark.parametrize(
    ('hidden_size', 'num_attention_heads', 'batch_size', 'seq_len'), [
        (30, 3, 8, 12),
        (4, 4, 4, 4)
    ]
)
def test_multi_head(hidden_size, num_attention_heads, batch_size, seq_len):
    config = HF_CONFIG(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        attention_probs_dropout_prob=0,
        output_attentions=False,
        layer_norm_eps=1.0,
        hidden_dropout_prob=0
    )

    inp = torch.rand(batch_size, seq_len, hidden_size)

    seed_everything(228)
    hf_att = BertAttention(config)
    hf_out = hf_att(hidden_states=inp)[0]

    seed_everything(228)
    att = MultiHeadMaskedAttention(
        n_heads=num_attention_heads,
        dim=hidden_size,
        layer_norm_eps=1.0
    )
    out = att(inp, inp, inp, None)

    assert torch.allclose(hf_out, out, rtol=1e-3)
