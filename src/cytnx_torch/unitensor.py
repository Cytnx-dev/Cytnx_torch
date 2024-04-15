from dataclasses import dataclass, field
from beartype.typing import List, Tuple
from abc import abstractmethod
import numpy as np
import torch

from .bond import AbstractBond, SymBond, Bond


# traits
@dataclass
class AbstractUniTensor:
    labels: List[str]
    bonds: List[AbstractBond]
    backend_args: dict = field(default_factory=dict)

    def _permute_meta(self, *args) -> Tuple[List[str], List[AbstractBond]]:
        args = list(args)
        new_labels = np.array(self.labels)[args]
        new_bonds = np.array(self.bonds)[args]
        return new_labels, new_bonds

    @property
    def rank(self) -> int:
        return len(self.labels)

    @property
    def shape(self) -> List[int]:
        return [b.dim for b in self.bonds]

    @property
    @abstractmethod
    def is_sym(self) -> bool:
        raise NotImplementedError("not implement for abstract type trait.")

    @abstractmethod
    def permute(self, *args, by_label: bool = True) -> "AbstractUniTensor":
        raise NotImplementedError("not implement for abstract type trait.")


@dataclass
class RegularUniTensor(AbstractUniTensor):

    data: torch.Tensor = field(default=None)

    def __post_init__(self):
        # check here, and also initialize torch tensor
        if self.data is None:
            self.data = torch.zeros(
                size=[b.dim for b in self.bonds], **self.backend_args
            )

        pass

    @property
    def is_sym(self) -> bool:
        return False

    def permute(self, *args, by_label: bool = True) -> "RegularUniTensor":

        if by_label:
            args = [self.labels.index(lbl) for lbl in args]

        new_labels, new_bonds = self._permute_meta(*args)
        return RegularUniTensor(
            labels=new_labels,
            bonds=new_bonds,
            backend_args=self.backend_args,
            data=self.data.permute(*args),
        )


@dataclass
class BlockUniTensor(AbstractUniTensor):

    blocks: List[torch.Tensor] = field(default_factory=list)

    def __post_init__(self):
        # check here, and also initialize torch tensor
        pass

    @property
    def is_sym(self) -> bool:
        return True

    def permute(self, *args, by_label: bool = True) -> "BlockUniTensor":

        if by_label:
            args = [self.labels.index(lbl) for lbl in args]

        new_labels, new_bonds = self._permute_meta(*args)
        new_blocks = [x.permute(*args) for x in self.blocks]
        return BlockUniTensor(
            labels=new_labels,
            bonds=new_bonds,
            backend_args=self.backend_args,
            blocks=new_blocks,
        )


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
