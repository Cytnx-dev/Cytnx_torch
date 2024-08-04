from dataclasses import dataclass
from beartype.typing import List
import torch

from ..bond import AbstractBond, SymBond, Bond
from .regular_unitensor import RegularUniTensor
from .block_unitensor import BlockUniTensor


# User API:
@dataclass(init=False)
class UniTensor:

    def __new__(
        cls,
        labels: List[str],
        bonds: List[AbstractBond],
        is_diag: bool = False,
        **backend_args,
    ):
        # check:
        if len(labels) != len(bonds):
            raise ValueError(
                f"number of labels should be equal to number of bonds. got {len(labels)} and {len(bonds)}"
            )

        # NOTE
        # please keep the if statement order as RegularUniTensor -> BlockUniTensor -> ...
        # (RegularUniTensor always comes first, so that when labels and bonds are empty, it correctly produce a scalar tensor)
        if all([isinstance(b, Bond) for b in bonds]):
            # no symmetry:
            return RegularUniTensor(
                labels=labels, bonds=bonds, is_diag=is_diag, backend_args=backend_args
            )
        elif all([isinstance(b, SymBond) for b in bonds]):
            # symmetry:
            return BlockUniTensor(
                labels=labels, bonds=bonds, is_diag=is_diag, backend_args=backend_args
            )
        else:
            raise ValueError(
                "unsupported bond type or mixing Bond and SymBond when declaring UniTensor."
            )

    @classmethod
    def from_torch(
        cls, tensor: torch.Tensor, labels: List[str] = None, is_diag: bool = False
    ) -> RegularUniTensor:

        if is_diag:
            if tensor.dim() != 1:
                raise ValueError("diagonal tensor should have one dimension.")

            x = tensor.shape[0]
            if labels:
                if len(labels) != 2:
                    raise ValueError("diagonal tensor should have two labels.")

            else:
                labels = [f"b{x}", f"b1{x}"]

            bonds = [Bond(dim=x), Bond(dim=x)]
        else:

            if labels is None:
                labels = [f"b{i}" for i in tensor.shape]

            if len(labels) != tensor.dim():
                raise ValueError(
                    f"number of labels should be equal to number of dimensions. got {len(labels)} and {tensor.dim()}"
                )
            bonds = [Bond(dim=x) for x in tensor.shape]

        return RegularUniTensor(
            labels=labels, bonds=bonds, data=tensor, is_diag=is_diag
        )
