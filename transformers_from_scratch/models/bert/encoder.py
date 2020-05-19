from torch import nn

from transformers_from_scratch.core.encoder import Encoder
from transformers_from_scratch.models.bert.layers import (
    BertEmbeddings,
    BertLayer
)
from transformers_from_scratch.models.bert.structures import (
    BertEncoderInput,
    BertEncoderOutput,
    BertEncoderConfig
)


class BertEncoder(Encoder):

    @property
    def token_embedding_weights(self):
        return self._embeddings.token_embedding_weights

    def __init__(self, config: BertEncoderConfig):
        super().__init__()

        self.config = config

        self._embeddings = BertEmbeddings(
            n_pos=config.n_pos,
            n_types=config.n_types,
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            pad_token_id=config.pad_token_id,
            layer_norm_eps=config.layer_norm_eps
        )

        self._layers = nn.ModuleList()
        for i_layer in range(config.n_layers):
            layer = BertLayer(
                hidden_dim=config.hidden_dim,
                n_heads=config.n_heads,
                layer_norm_eps=config.layer_norm_eps,
                intermediate_dim=config.intermediate_dim
            )

            self._layers.append(layer)

    def forward(self, encoder_input: BertEncoderInput) -> BertEncoderOutput:
        hidden_states = self._embeddings(
            token_ids=encoder_input.token_ids,
            token_type_ids=encoder_input.token_type_ids,
            token_pos=encoder_input.token_pos
        )

        all_hidden_states = []
        all_attentions = []

        for layer in self._layers:
            hidden_states, attentions = layer(hidden_states)
            all_hidden_states.append(hidden_states)
            all_attentions.append(attentions)

        out = BertEncoderOutput(
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )

        return out
