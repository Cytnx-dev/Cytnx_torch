from dataclasses import dataclass, field
from beartype.typing import List, Optional, Union
import numpy as np
import torch

from ..bond import BondType
from .base import AbstractUniTensor


@dataclass
class BlockUniTensorMeta:
    qn_indices_map: np.ndarray[int]  # shape = [nblock, rank]

    def permute(self, idx_map: int) -> "BlockUniTensorMeta":
        return BlockUniTensorMeta(qn_indices_map=self.qn_indices_map[:, idx_map])


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

    @property
    def requires_grad(self) -> bool:
        return np.all([x.requires_grad for x in self.blocks])

    def requires_grad_(self, requires_grad: bool = True) -> "BlockUniTensor":

        for blk in self.blocks:
            blk.requires_grad_(requires_grad)

        return self

    def grad(self) -> "BlockUniTensor":
        new_blocks = []
        for blk in self.blocks:
            grad_data = blk.grad

            if grad_data is None:
                grad_data = torch.Tensor(size=self.shape)

            new_blocks.append(grad_data)

        return BlockUniTensor(**self._get_generic_meta(), blocks=new_blocks)

    def backward(self) -> None:
        for blk in self.blocks:
            blk.backward()

    def permute(
        self, *args: Union[str, int], by_label: bool = True
    ) -> "BlockUniTensor":

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
