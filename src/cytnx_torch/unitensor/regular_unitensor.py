from dataclasses import dataclass, field
from beartype.typing import List, Optional, Union
import torch

from ..bond import Bond, BondType
from ..converter import RegularUniTensorConverter
from .base import AbstractUniTensor


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
                out += "           /             \\     " + "\n"
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

        out += "           \\             /     " + "\n"
        out += "            -------------      " + "\n"
        return out

    def __getitem__(self, key) -> "RegularUniTensor":

        accessor = key
        if not isinstance(key, tuple):
            accessor = (key,)

        if len(accessor) != self.rank:
            accessor = [*accessor, *([None] * (self.rank - len(accessor)))]

        remain_indices = []
        for i, kitem in enumerate(accessor):
            if kitem is None or isinstance(kitem, slice):
                remain_indices.append(i)
            elif isinstance(kitem, int):
                continue
            else:
                raise ValueError("key should be either int or slice.")

        new_data = self.data[key]

        assert len(remain_indices) == len(new_data.shape), "ERR, shape mismatch."

        new_bonds = [
            Bond(dim=dim, bond_type=self.bonds[remain_indices[i]].bond_type)
            for i, dim in enumerate(new_data.shape)
        ]
        new_labels = [self.labels[i] for i in remain_indices]

        return RegularUniTensor(
            labels=new_labels,
            bonds=new_bonds,
            backend_args=self.backend_args,
            data=new_data,
        )

    @property
    def is_sym(self) -> bool:
        return False

    @property
    def is_contiguous(self) -> bool:
        return self.data.is_contiguous()

    def contiguous(self) -> "RegularUniTensor":
        return RegularUniTensor(**self._get_generic_meta(), data=self.data.contiguous())

    @property
    def requires_grad(self) -> bool:
        return self.data.requires_grad

    def requires_grad_(self, requires_grad: bool = True) -> "RegularUniTensor":
        self.data.requires_grad_(requires_grad)
        return self

    def grad(self) -> "RegularUniTensor":
        grad_data = self.data.grad

        if grad_data is None:
            grad_data = torch.Tensor(size=self.shape)

        return RegularUniTensor(**self._get_generic_meta(), data=grad_data)

    def backward(self) -> None:
        self.data.backward()

    def permute(
        self, *args: Union[str, int], by_label: bool = True
    ) -> "RegularUniTensor":

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

    def contract(
        self, rhs: Union["RegularUniTensor", "RegularUniTensorConverter"]
    ) -> "RegularUniTensor":
        match rhs:
            case RegularUniTensor():
                raise NotImplementedError("TODO")
            case RegularUniTensorConverter():
                return rhs._contract(is_lhs=False, utensor=self)
            case _:
                raise ValueError(
                    "rhs should be either RegularUniTensor or RegularUniTensorConverter."
                )

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype
