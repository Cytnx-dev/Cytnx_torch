from cytnx_torch.bond import Qs, Bond, SymBond, BondType
from cytnx_torch.symmetry import U1, Zn
from cytnx_torch.unitensor import UniTensor, RegularUniTensor, BlockUniTensor


def test_reg():

    b1 = Bond(dim=10, bond_type=BondType.IN)
    b2 = Bond(dim=20, bond_type=BondType.OUT)

    ut = UniTensor(labels=["a", "b"], bonds=[b1, b2], dtype=float)

    assert isinstance(ut, RegularUniTensor)
    assert ut.backend_args["dtype"] is float


def test_sym():

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
