from dataclasses import dataclass, field
from beartype.typing import List
from abc import abstractmethod
import torch

from .bond import AbstractBond, SymBond, Bond


# traits
@dataclass
class AbstractUniTensor:
    labels: List[str]
    bonds: List[AbstractBond]
    backend_args: dict = field(default_factory=dict)

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


@dataclass
class RegularUniTensor(AbstractUniTensor):

    data: torch.Tensor = field(default=torch.Tensor([]))

    def __post_init__(self):
        # check here, and also initialize torch tensor

        self.data = torch.zeros(size=[b.dim for b in self.bonds], **self.backend_args)

        pass

    @property
    def is_sym(self) -> bool:
        return False


@dataclass
class BlockUniTensor(AbstractUniTensor):

    blocks: List[torch.Tensor] = field(default_factory=list)

    def __post_init__(self):
        # check here, and also initialize torch tensor
        pass

    @property
    def is_sym(self) -> bool:
        return True


# User API:
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
