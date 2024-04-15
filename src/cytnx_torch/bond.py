import numpy as np
from dataclasses import dataclass, field
from beartype.typing import List
from enum import Enum
from .symmetry import Symmetry


class BondType(Enum):
    IN = -1
    OUT = 1
    NONE = 0


@dataclass
class Qs:
    values: np.ndarray[int]
    degs: int = field(default=1, init=False)

    def __rshift__(self, deg: int):
        self.degs = deg
        return self

    def num_qnums(self):
        return len(self.values)


# trait
@dataclass
class AbstractBond:
    dim: int
    bond_type: BondType = field(default=BondType.NONE)


@dataclass
class Bond(AbstractBond):

    def __post_init__(self):
        if self.dim < 0:
            raise ValueError(f"dim should be non-negative. got {self.dim}")


@dataclass(init=False)
class SymBond(AbstractBond):

    # The following only works for bond with symmetry:
    _degs: np.ndarray[int] = field(
        default_factory=lambda: np.ndarray(shape=(0,), dtype=np.int64)
    )
    _qnums: np.ndarray[int] = field(
        default_factory=lambda: np.ndarray(shape=(0, 0), dtype=np.int64)
    )
    _syms: List[Symmetry] = field(default_factory=list)

    def _check_qnums(self) -> None:
        for i, s in enumerate(self._syms):
            if not s.check_qnums(self._qnums[:, i]):
                raise ValueError(f"invalid qnum detected for {i}-th symmetry: {s}")

    def __init__(self, bond_type: BondType, qnums: List[Qs], syms: List[Symmetry]):

        if bond_type == BondType.NONE:
            raise ValueError("SymBond should have bond_type != NONE.")

        self._bond_type = bond_type

        nqnum = np.unique([x.num_qnums() for x in qnums]).flatten()
        if len(nqnum) != 1:
            raise ValueError(
                "all Qs in qnums should have the same number of quantum numbers."
            )

        if nqnum[0] != len(syms):
            raise ValueError(
                "number of symmetry should match the number of quantum numbers."
            )

        self._qnums = np.vstack([x.values for x in qnums])
        self._degs = np.array([x.degs for x in qnums])
        self._syms = list(syms)

        self.dim = np.sum(self._degs)

        # checking qnums are consistent!
        self._check_qnums()
