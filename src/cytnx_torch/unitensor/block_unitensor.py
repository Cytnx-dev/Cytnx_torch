from dataclasses import dataclass
from beartype.typing import List, Optional, Union, Tuple, Sequence
import numpy as np
import torch
from ..symmetry import Symmetry, Qid
from ..bond import BondType, SymBond
from .base import AbstractUniTensor
from functools import cached_property
from cytnx_torch.internal_utils import ALL_ELEMENTS


@dataclass
class BlockUniTensorMeta:
    qn_indices_map: np.ndarray[int]  # shape = [nblock, rank]

    def rank(self) -> int:
        return self.qn_indices_map.shape[1]

    def permute(self, idx_map: Sequence[int]) -> "BlockUniTensorMeta":
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

    def select(
        self, key: List[Union[None, List[int]]]
    ) -> Tuple["BlockUniTensorMeta", List[int]]:
        if len(key) != self.rank():
            raise ValueError(
                "key should have the same length as the rank of the tensor"
            )

        new_maps = []
        old_blk_id = []
        for blk_id, map in enumerate(self.qn_indices_map):
            # TODO optimize this
            new_mp = []
            for m, k in zip(map, key):
                if k is None:
                    new_mp.append(m)
                else:
                    if m in k:
                        new_mp.append(k.index(m))
                    else:
                        break
            if len(new_mp) != self.rank():
                continue
            new_maps.append(new_mp)
            old_blk_id.append(blk_id)
        return BlockUniTensorMeta(qn_indices_map=np.array(new_maps)), old_blk_id


@dataclass
class BlockGenerator:
    bonds: List[SymBond]
    meta: Optional[BlockUniTensorMeta] = None

    def __post_init__(self):
        if len(self.bonds) > 1 and not self.bonds[0].check_same_symmetry(
            *self.bonds[1:]
        ):
            raise ValueError("bonds have different symmetry")

        if self.meta is None:
            self.meta = self._generate_meta()

    @cached_property
    def _get_symmetries(self) -> Tuple[Symmetry]:
        if len(self.bonds) > 0:
            return self.bonds[0]._syms
        else:
            return tuple()

    def _generate_meta(self) -> np.ndarray[int]:
        qnindices = [np.arange(len(bd._qnums)) for bd in self.bonds]
        qn_indices_map = np.meshgrid(*qnindices)
        qn_indices_map = np.array([mp.flatten() for mp in qn_indices_map]).T

        # filter out the ones that are not allowed:
        qn_indices_map = np.array(
            [
                qn_indices
                for qn_indices in qn_indices_map
                if self._can_have_block(qn_indices)
            ]
        )

        return BlockUniTensorMeta(qn_indices_map=qn_indices_map)

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
        return np.allclose(net_qns, 0)

    def __iter__(self):
        self.cntr = 0
        return self

    def __next__(self):
        if self.cntr < len(self.meta.qn_indices_map):
            qn_indices = self.meta.qn_indices_map[self.cntr]
            self.cntr += 1

            return torch.zeros(
                [bd._degs[qidx] for bd, qidx in zip(self.bonds, qn_indices)]
            )

        else:
            raise StopIteration


@dataclass
class BlockUniTensor(AbstractUniTensor):

    blocks: List[torch.Tensor] = None
    meta: BlockUniTensorMeta = None

    def __post_init__(self):

        if self.meta is None:
            bg = BlockGenerator(bonds=self.bonds)
            self.meta = bg.meta
        else:
            bg = BlockGenerator(bonds=self.bonds, meta=self.meta)

        # if blocks is not None, we don't generate from bg, and should be done carefully by internal!
        if self.blocks is None:
            self.blocks = [block for block in bg]

    def __getitem__(
        self, key: Union[Tuple, List[Qid], int, Qid, slice]
    ) -> "BlockUniTensor":
        """
        if element in qid_accessor is None, then it means all elements

        """

        def collect_qids_per_rank(item):
            qidx = None
            match item:
                case int():
                    raise NotImplementedError("int key is not supported yet")
                case slice():
                    if not item == ALL_ELEMENTS:
                        raise NotImplementedError(
                            "slice key currently only support all-elements, i.e. ':'"
                        )
                case Qid(value):
                    qidx = [value]
                case list():
                    # check instance:
                    if not all([isinstance(x, Qid) for x in item]):
                        raise ValueError("list should contain only Qid for now")

                    qidx = [x.value for x in item]

                case _:
                    raise ValueError(
                        "key should be either int, slice, Qid, or list of Qid"
                    )

            return qidx

        qid_accessor = []  # [naxis, list of qid]
        if isinstance(key, tuple):
            for item in key:
                qid_accessor.append(collect_qids_per_rank(item))
        else:
            qid_accessor.append(collect_qids_per_rank(key))

        # pad the rest with None
        qid_accessor += [None] * (self.rank - len(qid_accessor))

        assert (
            len(qid_accessor) == self.rank
        ), "key should have the same length as the rank of the tensor"

        # TODO create new metas:
        new_labels = self.labels
        new_bonds = [
            bd.slice_by_qindices(qids) for qids, bd in zip(qid_accessor, self.bonds)
        ]
        print(qid_accessor)
        # filter out the block and qnindices:
        new_meta, selected_blk_ids = self.meta.select(qid_accessor)
        new_blocks = [self.blocks[blk_id] for blk_id in selected_blk_ids]

        return BlockUniTensor(
            labels=new_labels,
            bonds=new_bonds,
            backend_args=self.backend_args,
            name=self.name,
            rowrank=self.rowrank,
            blocks=new_blocks,
            meta=new_meta,
        )

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
