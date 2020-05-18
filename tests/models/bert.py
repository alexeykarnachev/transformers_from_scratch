import torch
from transformers import BertForPreTraining, BertConfig

from transformers_from_scratch.core.utils import seed_everything
from transformers_from_scratch.models.bert.encoder import BertEncoder
from transformers_from_scratch.models.bert.model import BertPreTrainingModel
from transformers_from_scratch.models.bert.structures import \
    BertEncoderConfig, BertEncoderInput


def test_model():
    hf_config = BertConfig(
        vocab_size=1000,
        hidden_size=100,
        num_attention_heads=2,
        intermediate_size=256,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=100,
        num_hidden_layers=1,
    )

    config = BertEncoderConfig(
        hidden_dim=hf_config.hidden_size,
        n_heads=hf_config.num_attention_heads,
        layer_norm_eps=hf_config.layer_norm_eps,
        intermediate_dim=hf_config.intermediate_size,
        n_layers=hf_config.num_hidden_layers,
        n_pos=hf_config.max_position_embeddings,
        n_types=hf_config.type_vocab_size,
        vocab_size=hf_config.vocab_size,
        pad_token_id=hf_config.pad_token_id
    )

    bs = 8
    seq_len = 12

    seed_everything(228)
    hf_model = BertForPreTraining(hf_config)
    token_ids = torch.randint(
        low=0, high=hf_config.vocab_size, size=(bs, seq_len)
    )
    clf_labels = torch.randint(
        low=0, high=2, size=(bs,)
    )
    hf_loss = hf_model(
        token_ids,
        masked_lm_labels=token_ids,
        next_sentence_label=clf_labels
    )[0]

    seed_everything(228)
    encoder = BertEncoder(config)

    model = BertPreTrainingModel(encoder=encoder)
    token_ids = torch.randint(
        low=0, high=hf_config.vocab_size, size=(bs, seq_len)
    )
    clf_labels = torch.randint(
        low=0, high=2, size=(bs,)
    )

    inp = BertEncoderInput(
        token_ids=token_ids,
        token_type_ids=None,
        token_pos=None
    )

    loss = model(inp, head_labels={'lm': token_ids, 'clf': clf_labels}).loss

    assert torch.allclose(hf_loss, loss)
