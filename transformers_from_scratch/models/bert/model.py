from transformers_from_scratch.core.model import Model
from transformers_from_scratch.models.bert.backbone import BertBackbone
from transformers_from_scratch.models.bert.functions import init_weights
from transformers_from_scratch.models.bert.heads import \
    BertNextSentencePredictionHead, BertLMPredictionHead


class BertPreTrainingModel(Model):
    def __init__(self, backbone: BertBackbone):
        heads = {
            'clf': BertNextSentencePredictionHead(
                hidden_dim=backbone.config.hidden_dim
            ),
            'lm': BertLMPredictionHead(
                hidden_dim=backbone.config.hidden_dim,
                vocab_size=backbone.config.vocab_size,
                layer_norm_eps=backbone.config.layer_norm_eps,
                token_emb_weights=backbone.token_embedding_weights
            )
        }

        super().__init__(backbone, heads)

        self.apply(init_weights)
