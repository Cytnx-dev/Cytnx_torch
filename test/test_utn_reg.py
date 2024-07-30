from cytnx_torch.bond import Bond, BondType
from cytnx_torch.unitensor.regular_unitensor import RegularUniTensor
from cytnx_torch.unitensor import UniTensor

import torch
import numpy as np


def test_reg_ut():

    b1 = Bond(dim=10, bond_type=BondType.IN)
    b2 = Bond(dim=20, bond_type=BondType.OUT)

    ut = UniTensor(labels=["a", "b"], bonds=[b1, b2], dtype=float)

    assert isinstance(ut, RegularUniTensor)
    assert ut.backend_args["dtype"] is float
    assert ut.data.dtype is torch.float64
    assert ut.data.shape[0] == 10
    assert ut.data.shape[1] == 20


def test_relabel_reg():
    b1 = Bond(dim=10, bond_type=BondType.IN)
    b2 = Bond(dim=20, bond_type=BondType.OUT)
    b3 = Bond(dim=30, bond_type=BondType.OUT)

    ut = UniTensor(labels=["a", "b", "c"], bonds=[b1, b2, b3], dtype=float)

    ut2 = ut.relabel(old_labels=["b", "c"], new_labels=["c", "x"])

    assert np.all(ut2.labels == ["a", "c", "x"])
    assert isinstance(ut2, RegularUniTensor)


def test_get_item():
    b1 = Bond(dim=10, bond_type=BondType.IN)
    b2 = Bond(dim=20, bond_type=BondType.OUT)
    b3 = Bond(dim=30, bond_type=BondType.OUT)

    ut = UniTensor(labels=["a", "b", "c"], bonds=[b1, b2, b3], dtype=float)

    x = ut[0, ::2]

    assert x.shape == (10, 30)
    assert x.labels == ["b", "c"]
    assert x.bonds[0].bond_type == BondType.OUT
    assert x.bonds[1].bond_type == BondType.OUT


def test_set_item():
    b1 = Bond(dim=10, bond_type=BondType.IN)
    b2 = Bond(dim=20, bond_type=BondType.OUT)
    b3 = Bond(dim=30, bond_type=BondType.OUT)

    ut = UniTensor(labels=["a", "b", "c"], bonds=[b1, b2, b3], dtype=float)

    ut[0, ::2] = torch.arange(300).reshape(10, 30).to(float)

    x = ut[0, ::2]

    assert x.shape == (10, 30)
    assert x.labels == ["b", "c"]
    assert x.bonds[0].bond_type == BondType.OUT
    assert x.bonds[1].bond_type == BondType.OUT
    assert torch.allclose(x.data, torch.arange(300).reshape(10, 30).to(float))


def test_scalar_item():
    ut = UniTensor(labels=[], bonds=[], dtype=float)

    assert isinstance(ut, RegularUniTensor)
    assert ut.item() == 0.0


def test_as_matrix():
    b1 = Bond(dim=2, bond_type=BondType.IN)
    b2 = Bond(dim=3, bond_type=BondType.OUT)
    b3 = Bond(dim=4, bond_type=BondType.OUT)
    b4 = Bond(dim=5, bond_type=BondType.IN)

    ut = UniTensor(labels=["a", "b", "c", "d"], bonds=[b1, b2, b3, b4], dtype=float)

    ut.rowrank = 2

    mat, cl, cr = ut.as_matrix()

    assert mat.shape == (6, 20)
    assert mat.labels == ["_aux_L_", "_aux_R_"]

    reconstructed_ut = cl.contract(mat).contract(cr)

    assert reconstructed_ut.labels == ["a", "b", "c", "d"]
    assert reconstructed_ut == ut
