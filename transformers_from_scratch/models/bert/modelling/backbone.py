from torch import nn

from transformers_from_scratch.core.modelling.backbone import Backbone
from transformers_from_scratch.models.bert.modelling.layers import (
    BertEmbeddings,
    BertLayer
)
from transformers_from_scratch.models.bert.modelling.structures import (
    BertBackboneInput,
    BertBackboneOutput,
    BertBackboneConfig
)


class BertBackbone(Backbone):

    @property
    def token_embedding_weights(self):
        return self._embeddings.token_embedding_weights

    def __init__(self, config: BertBackboneConfig):
        super().__init__()

        self.config = config

        self._embeddings = BertEmbeddings(
            n_pos=config.n_pos,
            n_types=config.n_types,
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            pad_token_id=config.pad_token_id,
            layer_norm_eps=config.layer_norm_eps,
            hidden_dropout=config.hidden_dropout
        )

        self._layers = nn.ModuleList()
        for i_layer in range(config.n_layers):
            layer = BertLayer(
                hidden_dim=config.hidden_dim,
                n_heads=config.n_heads,
                layer_norm_eps=config.layer_norm_eps,
                intermediate_dim=config.intermediate_dim,
                attention_probs_dropout=config.attention_probs_dropout,
                hidden_dropout=config.hidden_dropout
            )

            self._layers.append(layer)

    def forward(self, backbone_input: BertBackboneInput) -> BertBackboneOutput:
        hidden_states = self._embeddings(
            token_ids=backbone_input.token_ids,
            token_type_ids=backbone_input.token_type_ids,
            token_pos=backbone_input.token_pos
        )

        all_hidden_states = []
        all_attentions = []

        for layer in self._layers:
            hidden_states, attentions = layer(hidden_states)
            all_hidden_states.append(hidden_states)
            all_attentions.append(attentions)

        out = BertBackboneOutput(
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )

        return out
