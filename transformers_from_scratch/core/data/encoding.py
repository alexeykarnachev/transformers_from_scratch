from dataclasses import dataclass
from typing import List


@dataclass
class Encoding:
    """Base data container which represents encoded text."""
    token_ids: List[int]
