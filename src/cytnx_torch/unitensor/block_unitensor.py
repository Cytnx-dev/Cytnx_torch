from dataclasses import dataclass, field
from beartype.typing import List, Optional, Union, Tuple
import numpy as np
import torch
from ..symmetry import Symmetry
from ..bond import BondType, SymBond
from .base import AbstractUniTensor
from functools import cached_property


@dataclass
class BlockGenerator:
    bonds: List[SymBond]
    look_up: np.ndarray[int] = field(init=False)  # shape = [prod(shape), rank]

    def __post_init__(self):
        if len(self.bonds) > 1 and not self.bonds[0].check_same_symmetry(
            *self.bonds[1:]
        ):
            raise ValueError("bonds have different symmetry")

        self.look_up = self._generate_look_up()

    @cached_property
    def _get_symmetries(self) -> Tuple[Symmetry]:
        if len(self.bonds) > 0:
            return self.bonds[0]._syms
        else:
            return tuple()

    def _generate_look_up(self) -> np.ndarray[int]:
        qnindices = [np.arange(len(bd._qnums)) for bd in self.bonds]
        qn_indices_map = np.meshgrid(*qnindices)
        qn_indices_map = np.array([mp.flatten() for mp in qn_indices_map]).T

        return qn_indices_map

    def _can_have_block(self, qn_indices: np.ndarray[int]) -> bool:
        # [rank, nsym]
        qns = np.array(
            [
                bd.get_qnum(qidx, directional=True)
                for bd, qidx in zip(self.bonds, qn_indices)
            ]
        )

        net_qns = [
            sym.merge_qnums(qns[:, i]) for i, sym in enumerate(self._get_symmetries)
        ]

        return np.all(net_qns == 0)

    def __iter__(self):
        self.cntr = 0
        return self

    def __next__(self):
        if self.cntr < len(self.look_up):
            qn_indices = self.look_up[self.cntr]
            self.cntr += 1

            if self._can_have_block(qn_indices):
                return qn_indices, torch.zeros(
                    [bd._degs[qidx] for bd, qidx in zip(self.bonds, qn_indices)]
                )
            else:
                return None, None
        else:
            raise StopIteration


@dataclass
class BlockUniTensorMeta:
    qn_indices_map: np.ndarray[int]  # shape = [nblock, rank]

    def rank(self) -> int:
        return self.qn_indices_map.shape[1]

    def permute(self, idx_map: int) -> "BlockUniTensorMeta":
        return BlockUniTensorMeta(qn_indices_map=self.qn_indices_map[:, idx_map])

    def get_block_idx(self, qn_indices: np.ndarray[int]) -> int | None:
        if len(qn_indices) != self.rank():
            raise ValueError(
                "qn_indices should have the same length as the rank of the tensor"
            )

        loc = np.where(np.all(self.qn_indices_map == qn_indices, axis=1)).flatten()

        if len(loc) == 0:
            return None
        else:
            return loc[0]


@dataclass
class BlockUniTensor(AbstractUniTensor):

    blocks: List[torch.Tensor] = None
    meta: Optional[BlockUniTensorMeta] = None

    def __post_init__(self):

        if self.meta is None and self.blocks is None:  # recalculate meta
            bg = BlockGenerator(bonds=self.bonds)

            blocks = []
            qn_indices_map = []
            for qn_indices, block in bg:
                if qn_indices is not None:
                    blocks.append(block)
                    qn_indices_map.append(qn_indices)
            self.blocks = blocks
            self.meta = BlockUniTensorMeta(qn_indices_map=np.array(qn_indices_map))

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
            meta=self.meta,
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

        return BlockUniTensor(
            **self._get_generic_meta(), blocks=new_blocks, meta=self.meta
        )

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
            meta=self.meta.permute(args),
        )

    def relabel(self, old_labels: List[str], new_labels: List[str]) -> "BlockUniTensor":

        new_ut = BlockUniTensor(
            **self._get_generic_meta(), blocks=self.blocks, meta=self.meta  # no clone
        )

        new_ut._relabel(old_labels, new_labels)
        return new_ut

    def to(
        self, device: torch.device = None, dtype: Optional[torch.dtype] = None
    ) -> "AbstractUniTensor":
        return BlockUniTensor(
            **self._get_generic_meta(),
            blocks=[x.to(device=device, dtype=dtype) for x in self.blocks],
            meta=self.meta,
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
