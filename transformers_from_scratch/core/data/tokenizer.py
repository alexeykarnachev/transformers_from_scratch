import abc

import tokenizers
from tokenizers.implementations import BaseTokenizer

from transformers_from_scratch.core.data.encoding import Encoding
from transformers_from_scratch.core.data.text_input import TextInput


class Tokenizer(BaseTokenizer, abc.ABC):
    def __init__(self, tokenizer: tokenizers.Tokenizer):
        super().__init__(tokenizer)

    @abc.abstractmethod
    def encode_for_train(self, text_input: TextInput) -> Encoding:
        pass
