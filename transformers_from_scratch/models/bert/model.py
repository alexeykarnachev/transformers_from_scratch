import torch

from transformers_from_scratch.core.heads import TokenClassificationHead
from transformers_from_scratch.core.model import Model
from transformers_from_scratch.models.bert.encoder import BertEncoder
from transformers_from_scratch.models.bert.heads import BertNextSentencePredictionHead


class BertPreTrainingModel(Model):
    def __init__(self, encoder: BertEncoder):
        heads = {
            'lm': TokenClassificationHead(
                hidden_dim=encoder.config.hidden_dim,
                n_classes=encoder.config.vocab_size
            ),
            'clf': BertNextSentencePredictionHead(
                hidden_dim=encoder.config.hidden_dim
            )
        }

        super().__init__(encoder, heads)

        self._tie_weights()

    def _tie_weights(self):
        o = self._heads['lm']._linear
        i = self._encoder._embeddings._token_emb

        if getattr(o, "bias", None) is not None:
            o.bias.data = torch.nn.functional.pad(
                o.bias.data,
                (0, o.weight.shape[0] - o.bias.shape[0],),
                "constant",
                0,
            )
        if hasattr(o, "out_features") and hasattr(i, "num_embeddings"):
            o.out_features = i.num_embeddings
