import pytest
import torch
import transformers

from transformers_from_scratch.core.utils import seed_everything
from transformers_from_scratch.models.bert import BertLayer


@pytest.mark.parametrize(
    'dim,n_heads,intermediate_dim', [
        (128, 4, 256)
    ]
)
def test_bert_block(dim,n_heads,intermediate_dim):
    hf_config = transformers.BertConfig(
        hidden_size=dim,
        num_attention_heads=n_heads,
        intermediate_size=intermediate_dim,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-12
    )

    seed_everything(228)
    inp = torch.rand(8, 12, 128)
    hf_bert_layer = transformers.BertLayer(hf_config)
    hf_out = hf_bert_layer(inp)[0]

    seed_everything(228)
    inp = torch.rand(8, 12, 128)
    bert_layer = BertLayer(
        dim=dim,
        n_heads=n_heads,
        layer_norm_eps=1e-12,
        intermediate_dim=intermediate_dim
    )
    out = bert_layer(inp)

    assert torch.allclose(hf_out, out, rtol=1e-3)