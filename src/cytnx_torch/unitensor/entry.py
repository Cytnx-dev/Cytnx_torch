from dataclasses import dataclass
from beartype.typing import List
import torch

from ..bond import AbstractBond, SymBond, Bond
from .regular_unitensor import RegularUniTensor
from .block_unitensor import BlockUniTensor


# User API:
@dataclass(init=False)
class UniTensor:

    def __new__(cls, labels: List[str], bonds: List[AbstractBond], **backend_args):
        # check:
        if len(labels) != len(bonds):
            raise ValueError(
                f"number of labels should be equal to number of bonds. got {len(labels)} and {len(bonds)}"
            )

        if all([isinstance(b, SymBond) for b in bonds]):
            # symmetry:
            return BlockUniTensor(labels=labels, bonds=bonds, backend_args=backend_args)

        elif all([isinstance(b, Bond) for b in bonds]):
            # no symmetry:
            return RegularUniTensor(
                labels=labels, bonds=bonds, backend_args=backend_args
            )
        else:
            raise ValueError(
                "unsupported bond type or mixing Bond and SymBond when declaring UniTensor."
            )

    @classmethod
    def from_torch(
        cls, tensor: torch.Tensor, labels: List[str] = None
    ) -> RegularUniTensor:

        if labels is None:
            labels = [f"b{i}" for i in tensor.shape]

        if len(labels) != tensor.dim():
            raise ValueError(
                f"number of labels should be equal to number of dimensions. got {len(labels)} and {tensor.dim()}"
            )

        return RegularUniTensor(
            labels=labels, bonds=[Bond(dim=x) for x in tensor.shape], data=tensor
        )
