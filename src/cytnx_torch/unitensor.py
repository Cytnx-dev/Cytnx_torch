from dataclasses import dataclass
from beartype.typing import List
from .bond import AbstractBond


@dataclass
class AbstractUniTensor:
    labels: List[str]
    bonds: List[AbstractBond]
