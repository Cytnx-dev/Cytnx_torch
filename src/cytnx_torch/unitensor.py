from dataclasses import dataclass, field
from beartype.typing import List, Tuple, Dict, Optional
from typing import Any
from abc import abstractmethod
import numpy as np
import torch

from .bond import AbstractBond, SymBond, Bond, BondType


# traits
@dataclass
class AbstractUniTensor:
    labels: List[str]
    bonds: List[AbstractBond]
    backend_args: dict = field(default_factory=dict)
    name: str = field(default="")
    rowrank: int = field(default=0)

    def _get_permuted_meta(self, *args) -> Tuple[List[str], List[AbstractBond]]:
        args = list(args)
        new_labels = np.array(self.labels)[args]
        new_bonds = np.array(self.bonds)[args]
        return new_labels, new_bonds

    def _get_generic_meta(self) -> Dict[str, Any]:
        return {
            "labels": self.labels,
            "bonds": self.bonds,
            "backend_args": self.backend_args,
            "name": self.name,
            "rowrank": self.rowrank,
        }

    @property
    def rank(self) -> int:
        return len(self.labels)

    @property
    def shape(self) -> Tuple[int]:
        return tuple([b.dim for b in self.bonds])

    @property
    @abstractmethod
    def is_contiguous(self) -> bool:
        raise NotImplementedError("not implement for abstract type trait.")

    @abstractmethod
    def contiguous(self) -> "AbstractUniTensor":
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
    def permute(self, *args, by_label: bool = True) -> "AbstractUniTensor":
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

    def _repr_body_diagram(self) -> str:
        Nin = self.rowrank
        Nout = self.rank - self.rowrank
        if Nin > Nout:
            vl = Nin
        else:
            vl = Nout

        out = "            -------------      " + "\n"
        for i in range(vl):
            if i == 0:
                out += "           /             \     " + "\n"
            else:
                out += "           |             |     " + "\n"

            if i < Nin:
                bks = BondType.get_symbol(self.bonds[i].bond_type, left_side=True)
                ls = "%3s %s__" % (self.labels[i], bks)
                llbl = "%-3d" % self.bonds[i].dim
            else:
                ls = "        "
                llbl = "   "
            if i < Nout:
                bks = BondType.get_symbol(
                    self.bonds[Nin + i].bond_type, left_side=False
                )
                r = "__%s %-3s" % (bks, self.labels[Nin + i])
                rlbl = "%3d" % self.bonds[Nin + i].dim
            else:
                r = "        "
                rlbl = "   "
            out += "   %s| %s     %s |%s" % (ls, llbl, rlbl, r) + "\n"

        out += "           \             /     " + "\n"
        out += "            -------------      " + "\n"
        return out

    @property
    def is_sym(self) -> bool:
        return False

    @property
    def is_contiguous(self) -> bool:
        return self.data.is_contiguous()

    def contiguous(self) -> "RegularUniTensor":
        return RegularUniTensor(**self._get_generic_meta(), data=self.data.contiguous())

    def permute(self, *args, by_label: bool = True) -> "RegularUniTensor":

        if by_label:
            args = [self.labels.index(lbl) for lbl in args]

        new_labels, new_bonds = self._get_permuted_meta(*args)
        return RegularUniTensor(
            labels=new_labels,
            bonds=new_bonds,
            backend_args=self.backend_args,
            data=self.data.permute(*args),
        )

    def relabel(
        self, old_labels: List[str], new_labels: List[str]
    ) -> "RegularUniTensor":

        new_ut = RegularUniTensor(
            **self._get_generic_meta(), data=self.data  # no clone
        )

        new_ut._relabel(old_labels, new_labels)
        return new_ut

    def to(
        self, device: torch.device = None, dtype: Optional[torch.dtype] = None
    ) -> "AbstractUniTensor":
        return RegularUniTensor(
            **self._get_generic_meta(), data=self.data.to(device=device, dtype=dtype)
        )

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype


@dataclass
class BlockUniTensor(AbstractUniTensor):

    blocks: List[torch.Tensor] = field(default_factory=list)

    def __post_init__(self):
        # check here, and also initialize torch tensor
        pass

    def _repr_body_diagram(self) -> str:
        Nin = self.rowrank
        Nout = self.rank - self.rowrank
        if Nin > Nout:
            vl = Nin
        else:
            vl = Nout

        out = "      |in>               <out| " + "\n"
        out += "           ---------------      " + "\n"
        for i in range(vl):
            out += "           |             |     " + "\n"
            if i < Nin:
                bks = BondType.get_symbol(self.bonds[i].bond_type, left_side=True)
                ls = "%3s %s__" % (self.labels[i], bks)
                llbl = "%-3d" % self.bonds[i].dim
            else:
                ls = "        "
                llbl = "   "
            if i < Nout:
                bks = BondType.get_symbol(
                    self.bonds[Nin + i].bond_type, left_side=False
                )
                r = "__%s %-3s" % (bks, self.labels[Nin + i])
                rlbl = "%3d" % self.bonds[Nin + i].dim
            else:
                r = "        "
                rlbl = "   "
            out += "   %s| %s     %s |%s" % (ls, llbl, rlbl, r) + "\n"
        out += "           |             |     " + "\n"
        out += "           ---------------     " + "\n"
        return out

    @property
    def is_sym(self) -> bool:
        return True

    @property
    def is_contiguous(self) -> bool:
        return np.all([blk.is_contiguous() for blk in self.blocks])

    def contiguous(self) -> "BlockUniTensor":
        return BlockUniTensor(
            **self._get_generic_meta(),
            blocks=[blk.contiguous() for blk in self.blocks],
        )

    def permute(self, *args, by_label: bool = True) -> "BlockUniTensor":

        if by_label:
            args = [self.labels.index(lbl) for lbl in args]

        new_labels, new_bonds = self._get_permuted_meta(*args)
        new_blocks = [x.permute(*args) for x in self.blocks]
        return BlockUniTensor(
            labels=new_labels,
            bonds=new_bonds,
            backend_args=self.backend_args,
            blocks=new_blocks,
        )

    def relabel(self, old_labels: List[str], new_labels: List[str]) -> "BlockUniTensor":

        new_ut = BlockUniTensor(
            **self._get_generic_meta(), blocks=self.blocks  # no clone
        )

        new_ut._relabel(old_labels, new_labels)
        return new_ut

    def to(
        self, device: torch.device = None, dtype: Optional[torch.dtype] = None
    ) -> "AbstractUniTensor":
        return BlockUniTensor(
            **self._get_generic_meta(),
            blocks=[x.to(device=device, dtype=dtype) for x in self.blocks],
        )

    @property
    def device(self) -> torch.device:
        if len(self.blocks) == 0:
            return torch.device("cpu")
        else:
            return self.blocks[0].device

    @property
    def dtype(self) -> torch.dtype:
        if len(self.blocks) == 0:
            return torch.get_default_dtype()
        else:
            return self.blocks[0].data.dtype


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
