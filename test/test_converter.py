import numpy as np
from cytnx_torch.bond import Bond, BondType
from cytnx_torch.converter import RegularUniTensorConverter
from cytnx_torch.unitensor.regular_unitensor import RegularUniTensor
from cytnx_torch.unitensor import UniTensor


def test_revert():

    b1 = Bond(dim=2)
    b2 = Bond(dim=4)
    b3 = Bond(dim=6)
    b4 = Bond(dim=5)

    rc1 = RegularUniTensorConverter(
        input_bonds=[b1, b2],
        input_labels=["a", "b"],
        output_bonds=[b3, b4],
        output_labels=["c", "d"],
    )

    rc2 = rc1.revert()

    assert isinstance(rc2, RegularUniTensorConverter)

    assert np.all(rc2.input_bonds == [b3, b4])
    assert np.all(rc2.input_labels == ["c", "d"])
    assert np.all(rc2.output_bonds == [b1, b2])
    assert np.all(rc2.output_labels == ["a", "b"])

    # check no copy when Bond
    assert np.all([rc2.input_bonds[i] is rc1.output_bonds[i] for i in range(2)])
    assert np.all([rc2.output_bonds[i] is rc1.input_bonds[i] for i in range(2)])


def test_revert_direct():

    b1 = Bond(dim=2, bond_type=BondType.IN)
    b2 = Bond(dim=4, bond_type=BondType.IN)
    b3 = Bond(dim=6, bond_type=BondType.OUT)
    b4 = Bond(dim=5, bond_type=BondType.OUT)

    rc1 = RegularUniTensorConverter(
        input_bonds=[b1, b2],
        input_labels=["a", "b"],
        output_bonds=[b3, b4],
        output_labels=["c", "d"],
    )

    rc2 = rc1.revert()

    assert isinstance(rc2, RegularUniTensorConverter)

    assert np.all(rc2.input_bonds == [b3.redirect(), b4.redirect()])
    assert np.all(rc2.input_labels == ["c", "d"])
    assert np.all(rc2.output_bonds == [b1.redirect(), b2.redirect()])
    assert np.all(rc2.output_labels == ["a", "b"])


def test_contract_ut_lhs():

    b1 = Bond(dim=2, bond_type=BondType.IN)
    b2 = Bond(dim=4, bond_type=BondType.IN)
    b3 = Bond(dim=6, bond_type=BondType.OUT)
    b4 = Bond(dim=5, bond_type=BondType.OUT)

    b34 = Bond(dim=5 * 6, bond_type=BondType.OUT)

    ut = UniTensor(labels=["a", "b", "c", "d"], bonds=[b1, b2, b3, b4], dtype=float)

    rc1 = RegularUniTensorConverter(
        input_bonds=[b3.redirect(), b4.redirect()],
        input_labels=["c", "d"],
        output_bonds=[b34],
        output_labels=["e"],
    )

    out = rc1.contract(ut)

    assert isinstance(out, RegularUniTensor)
    assert np.all(out.labels == ["e", "a", "b"])
    assert np.all(out.bonds[0] == b34)
    assert np.all(out.shape == (30, 2, 4))


def test_contract_ut_rhs():

    b1 = Bond(dim=2, bond_type=BondType.IN)
    b2 = Bond(dim=4, bond_type=BondType.IN)
    b3 = Bond(dim=6, bond_type=BondType.OUT)
    b4 = Bond(dim=5, bond_type=BondType.OUT)

    b34 = Bond(dim=5 * 6, bond_type=BondType.OUT)

    ut = UniTensor(labels=["a", "b", "c", "d"], bonds=[b1, b2, b3, b4], dtype=float)

    rc1 = RegularUniTensorConverter(
        input_bonds=[b3.redirect(), b4.redirect()],
        input_labels=["c", "d"],
        output_bonds=[b34],
        output_labels=["e"],
    )

    out = ut.contract(rc1)

    assert isinstance(out, RegularUniTensor)
    assert np.all(out.labels == ["a", "b", "e"])
    assert np.all(out.bonds[-1] == b34)
    assert np.all(out.shape == (2, 4, 30))
