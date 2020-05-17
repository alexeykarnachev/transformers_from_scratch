import pytest
import torch
from transformers import BertConfig as HFBertConfig
from transformers.modeling_bert import BertEmbeddings as HFBertEmbeddings
from transformers.modeling_bert import BertLayer as HFBertLayer

from transformers_from_scratch.core.utils import seed_everything
from transformers_from_scratch.models.bert import (
    BertLayer,
    BertEmbeddings
)


@pytest.mark.parametrize(
    'dim,n_heads,intermediate_dim', [
        (128, 4, 256), (4, 4, 4), (128, 1, 128)
    ]
)
def test_bert_block(dim, n_heads, intermediate_dim):
    bs = 8
    seq_len = 12

    hf_config = HFBertConfig(
        hidden_size=dim,
        num_attention_heads=n_heads,
        intermediate_size=intermediate_dim,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-12
    )

    seed_everything(228)
    inp = torch.rand(bs, seq_len, dim)
    hf_bert_layer = HFBertLayer(hf_config)
    hf_out = hf_bert_layer(inp)[0]

    seed_everything(228)
    inp = torch.rand(bs, seq_len, dim)
    bert_layer = BertLayer(
        dim=dim,
        n_heads=n_heads,
        layer_norm_eps=1e-12,
        intermediate_dim=intermediate_dim
    )
    out = bert_layer(inp)

    assert torch.allclose(hf_out, out, rtol=1e-3)


@pytest.mark.parametrize(
    'dim,vocab_size', [
        (10, 100), (100, 100), (5, 5), (1, 1), (3, 3)
    ]
)
def test_bert_embeddings(dim, vocab_size):
    bs = 8
    seq_len = 12

    hf_config = HFBertConfig(
        hidden_size=dim,
        hidden_dropout_prob=0.0,
        pad_token_id=0,
        vocab_size=vocab_size,
        max_position_embeddings=seq_len,
        type_vocab_size=2,
        layer_norm_eps=1e-12
    )

    seed_everything(228)
    inp = torch.randint(low=0, high=vocab_size, size=(bs, seq_len))
    hf_bert_embeddings = HFBertEmbeddings(hf_config)
    hf_out = hf_bert_embeddings(inp)

    seed_everything(228)
    inp = torch.randint(low=0, high=vocab_size, size=(bs, seq_len))
    bert_embeddings = BertEmbeddings(
        dim=dim,
        n_pos=seq_len,
        vocab_size=vocab_size,
        n_types=2,
        pad_token_id=0,
        layer_norm_eps=1e-12
    )
    out = bert_embeddings(inp)

    assert torch.allclose(hf_out, out, rtol=1e-3)
