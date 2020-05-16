from collections import namedtuple

import pytest
import torch
from transformers.modeling_bert import BertSelfAttention

from transformers_from_scratch.core.attention import MultiHeadAttention
from transformers_from_scratch.core.utils import seed_everything

HF_CONFIG = namedtuple(
    typename='Config',
    field_names=(
        'hidden_size',
        'num_attention_heads',
        'attention_probs_dropout_prob',
        'output_attentions'
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
        output_attentions=False
    )

    inp = torch.rand(batch_size, seq_len, hidden_size)

    seed_everything(228)
    hf_att = BertSelfAttention(config)
    hf_out = hf_att(hidden_states=inp)[0]

    seed_everything(228)
    att = MultiHeadAttention(n_heads=num_attention_heads, dim=hidden_size)
    out = att(inp, inp, inp)

    assert torch.allclose(hf_out, out)
