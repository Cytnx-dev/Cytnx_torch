import numpy as np
from dataclasses import dataclass, field
from beartype.typing import List
from enum import Enum
from abc import abstractmethod
from .symmetry import Symmetry


class BondType(Enum):
    IN = -1
    OUT = 1
    NONE = 0

    @staticmethod
    def get_symbol(bond_type: "BondType", left_side: bool) -> str:
        if left_side:
            match bond_type:
                case BondType.IN:
                    bks = "> "
                case BondType.OUT:
                    bks = "<*"
                case BondType.NONE:
                    bks = "__"
                case _:
                    raise ValueError("invalid bond type")
        else:
            print("right", bond_type)
            match bond_type:
                case BondType.IN:
                    bks = "<*"
                case BondType.OUT:
                    bks = " >"
                case BondType.NONE:
                    bks = "__"
                case _:
                    raise ValueError("invalid bond type")
        return bks


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

    @property
    @abstractmethod
    def nsym(self) -> int:
        raise NotImplementedError("not implement for abstract type trait.")


@dataclass
class Bond(AbstractBond):

    def __post_init__(self):
        if self.dim < 0:
            raise ValueError(f"dim should be non-negative. got {self.dim}")

    def __str__(self):
        out = f"Dim = {self.dim} |\n"
        out += f"{self.bond_type} :\n"
        return out

    @property
    def nsym(self) -> int:
        return 0


@dataclass(init=False)
class SymBond(AbstractBond):

    # The following only works for bond with symmetry:
    # For better performance we convert Qs to numpy array instead of using List[Qs]
    _degs: np.ndarray[int] = field(
        default_factory=lambda: np.ndarray(shape=(0,), dtype=np.int64)
    )
    _qnums: np.ndarray[int] = field(
        default_factory=lambda: np.ndarray(shape=(0, 0), dtype=np.int64)
    )
    _syms: List[Symmetry] = field(default_factory=list)

    @property
    def nsym(self) -> int:
        return len(self._syms)

    def _check_qnums(self) -> None:
        for i, s in enumerate(self._syms):
            if not s.check_qnums(self._qnums[:, i]):
                raise ValueError(f"invalid qnum detected for {i}-th symmetry: {s}")

    def __init__(self, bond_type: BondType, qnums: List[Qs], syms: List[Symmetry]):

        if bond_type == BondType.NONE:
            raise ValueError("SymBond should have bond_type != NONE.")

        self.bond_type = bond_type

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

    def __str__(self):

        out = f"Dim = {self.dim} |\n"
        out += f"{self.bond_type} :\n"

        for n in range(self.nsym):
            out += f" {str(self._syms[n])}:: "
            for idim in range(len(self._qnums)):
                out += " %+d" % (self._qnums[idim, n])
            out += "\n         "

        return out
