from cytnx_torch.bond import Qs, SymBond, BondType
from cytnx_torch.symmetry import U1, Zn
from cytnx_torch.unitensor import UniTensor
from cytnx_torch.unitensor.block_unitensor import BlockUniTensor, Qid
import numpy as np


def test_sym_ut():

    b1 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )
    b2 = SymBond(
        bond_type=BondType.OUT,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )

    ut = UniTensor(labels=["a", "b"], bonds=[b1, b2], dtype=float)

    assert isinstance(ut, BlockUniTensor)
    assert ut.backend_args["dtype"] is float
    for b in ut.bonds:
        assert not b.bond_type == BondType.NONE


def test_relabel_blk():
    b1 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )
    b2 = SymBond(
        bond_type=BondType.OUT,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )

    b3 = SymBond(
        bond_type=BondType.OUT,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )

    ut = UniTensor(labels=["a", "b", "c"], bonds=[b1, b2, b3], dtype=float)

    ut2 = ut.relabel(old_labels=["b", "c"], new_labels=["c", "x"])

    assert np.all(ut2.labels == ["a", "c", "x"])
    assert isinstance(ut2, BlockUniTensor)


def test_getitem():
    b1 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )
    b2 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )
    b3 = SymBond(
        bond_type=BondType.OUT,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 0]) >> 4, Qs([-1, 1]) >> 5],
        syms=[U1(), Zn(n=2)],
    )

    ut = UniTensor(labels=["a", "b", "c"], bonds=[b1, b2, b3], dtype=float)

    x = ut[Qid(0), :, [Qid(0), Qid(2)]]

    assert x.shape == (3, 7, 8)
    assert x.labels == ["a", "b", "c"]
    assert len(x.blocks) == 1
    assert x.meta.qn_indices_map.shape == (1, 3)
    assert np.all(x.meta.qn_indices_map[0] == [0, 1, 1])


def test_backend_arg_propogate():
    b1 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )
    b2 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )
    b3 = SymBond(
        bond_type=BondType.OUT,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 0]) >> 4, Qs([-1, 1]) >> 5],
        syms=[U1(), Zn(n=2)],
    )

    ut = UniTensor(labels=["a", "b", "c"], bonds=[b1, b2, b3], dtype=int)

    for blk in ut.blocks:
        blk.dtype == int
