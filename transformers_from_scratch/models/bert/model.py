from transformers_from_scratch.core.model import Model
from transformers_from_scratch.models.bert.encoder import BertEncoder
from transformers_from_scratch.models.bert.functions import init_weights
from transformers_from_scratch.models.bert.heads import \
    BertNextSentencePredictionHead, BertLMPredictionHead


class BertPreTrainingModel(Model):
    def __init__(self, encoder: BertEncoder):
        heads = {
            'clf': BertNextSentencePredictionHead(
                hidden_dim=encoder.config.hidden_dim
            ),
            'lm': BertLMPredictionHead(
                hidden_dim=encoder.config.hidden_dim,
                vocab_size=encoder.config.vocab_size,
                layer_norm_eps=encoder.config.layer_norm_eps,
                token_emb_weights=encoder.token_embedding_weights
            )
        }

        super().__init__(encoder, heads)

        self.apply(init_weights)
