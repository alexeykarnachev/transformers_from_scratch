import abc
from typing import List

from transformers_from_scratch.core.data.text_input import TextInput


class TextLinesParser:
    @abc.abstractmethod
    def parse(self, text: str) -> List[TextInput]:
        pass
