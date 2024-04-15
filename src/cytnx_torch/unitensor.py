from dataclasses import dataclass, field
from beartype.typing import List
from .bond import AbstractBond, SymBond, Bond


# traits
@dataclass
class AbstractUniTensor:
    labels: List[str]
    bonds: List[AbstractBond]
    backend_args: dict = field(default_factory=dict)


@dataclass
class RegularUniTensor(AbstractUniTensor):

    def __post_init__(self):
        # check here, and also initialize torch tensor
        pass


@dataclass
class BlockUniTensor(AbstractUniTensor):

    def __post_init__(self):
        # check here, and also initialize torch tensor
        pass


# driver:
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
