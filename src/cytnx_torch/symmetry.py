import numpy as np
from dataclasses import dataclass, field
from beartype.typing import Union
from abc import abstractmethod


@dataclass
class Qid:
    value: int


@dataclass(frozen=True)
class Symmetry:
    label: str = field(default="")

    @abstractmethod
    def combine_rule(
        self, A: Union[np.ndarray[int], int], B: Union[np.ndarray[int], int]
    ) -> Union[np.ndarray[int], int]:
        """combine two quantum numbers (or elementwise)"""
        raise NotImplementedError("not implement for abstract type trait.")

    @abstractmethod
    def reverse_rule(
        self, A: Union[np.ndarray[int], int]
    ) -> Union[np.ndarray[int], int]:
        """reverse a list of quantum numbers"""
        raise NotImplementedError("not implement for abstract type trait.")

    @abstractmethod
    def merge_rule(self, qns: np.ndarray[int]) -> int:
        """merge multiple qns all in once"""
        raise NotImplementedError("not implement for abstract type trait.")

    @abstractmethod
    def check_qnums(self, qnums: np.ndarray[int]) -> bool:
        raise NotImplementedError("not implement for abstract type trait.")

    def combine_qnums(
        self, qnums_a: np.ndarray[int], qnums_b: np.ndarray[int]
    ) -> np.ndarray[int]:
        mesh_b, mesh_a = np.meshgrid(qnums_b, qnums_a)
        return self.combine_rule(mesh_a.flatten(), mesh_b.flatten())

    def merge_qnums(self, qnums: np.ndarray[int]) -> int:
        return self.merge_rule(qnums)

    def reverse_qnums(self, qnums: np.ndarray[int]) -> np.ndarray[int]:
        return self.reverse_rule(np.array(qnums))


# trait
@dataclass(frozen=True)
class AbelianSym(Symmetry):
    pass

    def reverse_rule(
        self, A: Union[np.ndarray[int], int]
    ) -> Union[np.ndarray[int], int]:
        return -A


@dataclass(frozen=True)
class U1(AbelianSym):
    """
    Unitary Symmetry class.
    The U1 symmetry can have quantum number represent as arbitrary unsigned integer.

    Fusion rule for combine two quantum number:

        q1 + q2

    """

    def combine_rule(
        self, A: Union[np.ndarray[int], int], B: Union[np.ndarray[int], int]
    ) -> Union[np.ndarray[int], int]:
        return A + B

    def merge_rule(self, qns: np.ndarray[int]) -> int:
        return np.sum(qns)

    def __str__(self) -> str:
        return f"U1 label={self.label}"

    def check_qnums(self, qnums: np.ndarray[int]) -> bool:
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

    def merge_rule(self, qns: np.ndarray[int]) -> int:
        return np.sum(qns) % self.n

    def combine_rule(
        self, A: Union[np.ndarray[int], int], B: Union[np.ndarray[int], int]
    ) -> int:
        return (A + B) % self.n

    def __str__(self):
        return f"Z{self.n} label={self.label}"

    def check_qnums(self, qnums: np.ndarray[int]) -> bool:
        return np.all([(q >= 0) and (q < self.n) for q in qnums])
