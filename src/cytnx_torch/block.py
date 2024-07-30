from dataclasses import dataclass
from typing import List
import torch

from .bond import SymBond


@dataclass
class FragmentedBlock:
    data: torch.Tensor
    qindices: List[int]

    def __post_init__(self):
        assert len(self.qindices) == len(self.data.shape), "dimension mismatch"

    def get_dims(self, bonds: List[SymBond]) -> List[int]:
        assert len(self.qindices) == len(bonds), "bond dimension does not match"
        return [b._degs[self.qindices[i]] for i, b in enumerate(bonds)]
