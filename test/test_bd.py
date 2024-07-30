from cytnx_torch.bond import Qs, Bond, SymBond, BondType
from cytnx_torch.symmetry import U1, Zn


def test_Qs():

    q1 = Qs([1, 2, 3, 4]) >> 10

    assert q1.degs == 10


def test_bond():

    b1 = Bond(dim=10)

    b2 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )

    assert b1.bond_type == BondType.NONE
    assert b2.dim == 7
    assert not b1 == b2


def test_eq_reg_bond():
    b1 = Bond(dim=10)
    b2 = Bond(dim=10)

    assert b1 == b2


def test_eq_sym_bond():
    b2 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )

    b1 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )

    assert b1 == b2

    b3 = SymBond(
        bond_type=BondType.OUT,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )

    assert b1 != b3


def test_slice_by_qnid():
    b2 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4, Qs([3, 0]) >> 4, Qs([4, 1]) >> 5],
        syms=[U1(), Zn(n=2)],
    )

    b1 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([3, 0]) >> 4],
        syms=[U1(), Zn(n=2)],
    )

    b3 = b2.slice_by_qindices([0, 2])

    assert b3 == b1
    assert b3.dim == b1.dim


def test_contractible():
    b2 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )

    b1 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )

    assert not b1.contractable_with(b2)

    b3 = SymBond(
        bond_type=BondType.OUT,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )

    assert b1.contractable_with(b3)


def test_same_sym():
    b2 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )

    b1 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )
    b3 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )
    assert b1.check_same_symmetry(b2, b3)


def test_same_sym_invalid():
    b2 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )

    b1 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=4)],
    )
    b3 = SymBond(
        bond_type=BondType.IN,
        qnums=[Qs([-1, 0]) >> 3, Qs([0, 1]) >> 4],
        syms=[U1(), Zn(n=2)],
    )
    assert not b1.check_same_symmetry(b2, b3)
