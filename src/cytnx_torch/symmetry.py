import numpy as np
from dataclasses import dataclass, field
from beartype.typing import List
from abc import abstractmethod


@dataclass(frozen=True)
class Symmetry:
    label: str = field(default="")


# trait
@dataclass(frozen=True)
class AbelianSym(Symmetry):

    @abstractmethod
    def combine_rule(self, A: int, B: int) -> int:
        raise NotImplementedError("not implement for abstract type trait.")

    @abstractmethod
    def check_qnums(self, qnums: List[int]) -> bool:
        raise NotImplementedError("not implement for abstract type trait.")

    def combine_qnums(self, qnums_a: List[int], qnums_b: List[int]) -> List[int]:
        mesh_b, mesh_a = np.meshgrid(qnums_b, qnums_a)
        return self.combine_rule(mesh_a.flatten(), mesh_b.flatten())


@dataclass(frozen=True)
class U1(AbelianSym):
    """
    Unitary Symmetry class.
    The U1 symmetry can have quantum number represent as arbitrary unsigned integer.

    Fusion rule for combine two quantum number:

        q1 + q2

    """

    def combine_rule(self, A: int, B: int) -> int:
        return A + B

    def __str__(self) -> str:
        return f"U1 label={self.label}"

    def check_qnums(self, qnums: List[int]) -> bool:
        return True


@dataclass(frozen=True)
class Zn(AbelianSym):
    """
    Z(n) Symmetry class.
    The Z(n) symmetry can have integer quantum number, with n > 1.

    Fusion rule for combine two quantum number:

        (q1 + q2)%n
    """

    n: int = field(default=2)

    def __post_init__(self):
        if self.n < 2:
            raise ValueError(
                "Symmetry.Zn", "[ERROR] discrete symmetry Zn must have n >= 2."
            )

    def combine_rule(self, A: int, B: int) -> int:
        return (A + B) % self.n

    def __str__(self):
        return f"Z{self.n} label={self.label}"

    def check_qnums(self, qnums: List[int]) -> bool:
        return np.all([(q >= 0) and (q < self.n) for q in qnums])
