from cytnx_torch.bond import Qs, SymBond, BondType
from cytnx_torch.symmetry import U1, Zn
from cytnx_torch.unitensor.block_unitensor import BlockGenerator
import pytest


def test_generator_init():

    b1 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4, Qs([1, 1]) >> 5],
        syms=[U1(), Zn(n=2)],
    )
    b2 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )
    b3 = SymBond(
        bond_type=BondType.OUT,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )

    bg = BlockGenerator(bonds=[b1, b2, b3])

    expected_look_up = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
        [2, 0, 0],
        [2, 0, 1],
        [2, 1, 0],
        [2, 1, 1],
    ]

    assert bg.look_up.shape == (3 * 2 * 2, 3)

    assert set(tuple(x) for x in bg.look_up.tolist()) == set(
        tuple(x) for x in expected_look_up
    )


def test_invalid_generator_init():

    b1 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4, Qs([1, 1]) >> 5],
        syms=[U1(), Zn(n=2)],
    )
    b2 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=3)],
    )
    b3 = SymBond(
        bond_type=BondType.OUT,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )

    with pytest.raises(ValueError):
        BlockGenerator(bonds=[b1, b2, b3])


def test_generator_interator():

    b1 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4, Qs([1, 1]) >> 5],
        syms=[U1(), Zn(n=2)],
    )
    b2 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )
    b3 = SymBond(
        bond_type=BondType.OUT,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )

    bg = BlockGenerator(bonds=[b1, b2, b3])

    expected_look_up = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
        [2, 0, 0],
        [2, 0, 1],
        [2, 1, 0],
        [2, 1, 1],
    ]
    expected_look_up = set(tuple(x) for x in expected_look_up)

    for qn_indices, blk in bg:
        if qn_indices is not None:
            assert tuple(qn_indices) in expected_look_up
