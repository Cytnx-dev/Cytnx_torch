from dataclasses import dataclass, field
from beartype.typing import List, Tuple, Dict, Optional, Union
from typing import Any
from abc import abstractmethod
import numpy as np
import torch

from ..bond import AbstractBond
from ..converter import AbstractConverter


# traits
@dataclass
class AbstractUniTensor:
    labels: List[str]
    bonds: List[AbstractBond]
    backend_args: dict = field(default_factory=dict)
    is_diag: bool = field(default=False)
    name: str = field(default="")
    rowrank: int = field(default=0)

    def _get_permuted_meta(self, *args) -> Tuple[List[str], List[AbstractBond]]:
        args = list(args)
        new_labels = list(np.array(self.labels)[args])
        new_bonds = list(np.array(self.bonds)[args])
        return new_labels, new_bonds

    def _get_generic_meta(self) -> Dict[str, Any]:
        return {
            "labels": self.labels,
            "bonds": self.bonds,
            "backend_args": self.backend_args,
            "name": self.name,
            "rowrank": self.rowrank,
            "is_diag": self.is_diag,
        }

    def __eq__(self, rhs: "AbstractUniTensor") -> bool:
        return (
            self.labels == rhs.labels
            and self.bonds == rhs.bonds
            and self.name == rhs.name
            and self.rowrank == rhs.rowrank
        )

    @property
    def rank(self) -> int:
        return len(self.labels)

    @property
    def shape(self) -> Tuple[int]:
        return tuple([b.dim for b in self.bonds])

    @abstractmethod
    def __getitem__(self, key: Tuple) -> "AbstractUniTensor":
        raise NotImplementedError("not implement for abstract type trait.")

    @abstractmethod
    def __setitem__(self, key: Tuple) -> None:
        raise NotImplementedError("not implement for abstract type trait.")

    @abstractmethod
    def as_matrix(self) -> "AbstractUniTensor":
        raise NotImplementedError("not implement for abstract type trait.")

    @property
    @abstractmethod
    def is_contiguous(self) -> bool:
        raise NotImplementedError("not implement for abstract type trait.")

    @abstractmethod
    def contiguous(self) -> "AbstractUniTensor":
        raise NotImplementedError("not implement for abstract type trait.")

    @property
    @abstractmethod
    def requires_grad(self) -> bool:
        raise NotImplementedError("not implement for abstract type trait.")

    @abstractmethod
    def requires_grad_(self, requires_grad: bool = True) -> "AbstractUniTensor":
        raise NotImplementedError("not implement for abstract type trait.")

    @abstractmethod
    def grad(self) -> "AbstractUniTensor":
        raise NotImplementedError("not implement for abstract type trait.")

    @abstractmethod
    def backward(self) -> None:
        raise NotImplementedError("not implement for abstract type trait.")

    @abstractmethod
    def contract(
        self, rhs: Union["AbstractUniTensor", "AbstractConverter"]
    ) -> "AbstractUniTensor":
        raise NotImplementedError("not implement for abstract type trait.")

    def _relabel(self, old_labels: List[str], new_labels: List[str]) -> None:

        if len(old_labels) != len(new_labels):
            raise ValueError(
                f"old_labels: len={len(old_labels)} and new_labels: len={len(new_labels)} should have the same length."
            )

        out_labels = list(self.labels)
        idx = [self.labels.index(lbl) for lbl in old_labels]
        print(idx)

        for i, lbl in zip(idx, new_labels):
            out_labels[i] = lbl

        self.labels = out_labels

    @property
    @abstractmethod
    def is_sym(self) -> bool:
        raise NotImplementedError("not implement for abstract type trait.")

    @abstractmethod
    def permute(
        self, *args: Union[str, int], by_label: bool = True
    ) -> "AbstractUniTensor":
        raise NotImplementedError("not implement for abstract type trait.")

    @abstractmethod
    def relabel(
        self, old_labels: List[str], new_labels: List[str]
    ) -> "AbstractUniTensor":
        raise NotImplementedError("not implement for abstract type trait.")

    @abstractmethod
    def to(
        self, device: torch.device = None, dtype: Optional[torch.dtype] = None
    ) -> "AbstractUniTensor":
        raise NotImplementedError("not implement for abstract type trait.")

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError("not implement for abstract type trait.")

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        raise NotImplementedError("not implement for abstract type trait.")

    @abstractmethod
    def _repr_body_diagram(self) -> str:
        raise NotImplementedError("not implement for abstract type trait.")

    def get_label_index(self, label: str) -> int:
        return self.labels.index(label)

    def get_bond(self, id: Union[str, int]) -> AbstractBond:

        idx = None
        match id:
            case str():
                idx = self.labels.index(id)
            case int():
                idx = id
            case _:
                raise ValueError("id should be either str or int.")

        if idx >= len(self.bonds):
            raise ValueError(f"index {idx} is out of bound.")

        return self.bonds[idx]

    def print_diagram(self, is_bond_info=False) -> None:
        print("-----------------------")
        print(f"tensor Name : {self.name}")
        print(f"tensor Rank : {self.rank}")
        print(f"has_symmetry: {self.is_sym}")

        body_str = self._repr_body_diagram()
        print(body_str)

        if is_bond_info:
            for i in range(len(self.bonds)):
                print(f"lbl:{self.labels[i]} {str(self.bonds[i])}")
