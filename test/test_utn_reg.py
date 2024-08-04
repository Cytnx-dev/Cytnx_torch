from cytnx_torch.bond import Bond, BondType
from cytnx_torch.unitensor.regular_unitensor import RegularUniTensor
from cytnx_torch.unitensor import UniTensor

import torch
import numpy as np
import pytest


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

    mat, cl, cr = ut.as_matrix(left_bond_label="x", right_bond_label="y")

    assert mat.shape == (6, 20)
    assert mat.labels == ["x", "y"]

    reconstructed_ut = cl.contract(mat).contract(cr)

    assert reconstructed_ut.labels == ["a", "b", "c", "d"]
    assert reconstructed_ut == ut


def test_is_diag():
    b1 = Bond(dim=3, bond_type=BondType.IN)
    b2 = Bond(dim=3, bond_type=BondType.OUT)

    ut = UniTensor(labels=["a", "b"], bonds=[b1, b2], dtype=float, is_diag=True)

    assert ut.is_diag is True
    assert ut.data.shape == (3,)


def test_is_diag_invalid():
    b1 = Bond(dim=3, bond_type=BondType.IN)
    b2 = Bond(dim=2, bond_type=BondType.OUT)

    with pytest.raises(ValueError):
        UniTensor(labels=["a", "b"], bonds=[b1, b2], dtype=float, is_diag=True)


def test_init_from_torch_diag():
    x = torch.arange(6)

    ut = UniTensor.from_torch(x, labels=["a", "b"], is_diag=True)

    assert ut.is_diag is True
    assert ut.data.shape == (6,)


def test_init_from_torch_diag_invalid():
    x = torch.arange(6).reshape(2, 3)
    with pytest.raises(ValueError):
        UniTensor.from_torch(x, labels=["a", "b"], is_diag=True)


def test_getitem_diag():
    b1 = Bond(dim=3, bond_type=BondType.IN)
    b2 = Bond(dim=3, bond_type=BondType.OUT)

    ut = UniTensor(labels=["a", "b"], bonds=[b1, b2], dtype=float, is_diag=True)

    x = ut[:2, :2]

    assert x.is_diag is True
    assert x.data.shape == (2,)


def test_getitem_diag_non_diag_access():
    b1 = Bond(dim=3, bond_type=BondType.IN)
    b2 = Bond(dim=3, bond_type=BondType.OUT)

    ut = UniTensor(labels=["a", "b"], bonds=[b1, b2], dtype=float, is_diag=True)

    x = ut[:2]

    assert x.is_diag is False
    assert x.data.shape == (2, 3)


def test_grad():
    b1 = Bond(dim=3, bond_type=BondType.IN)
    b2 = Bond(dim=3, bond_type=BondType.OUT)

    ut = UniTensor(labels=["a", "b"], bonds=[b1, b2], dtype=float, is_diag=True)

    grad_ut = ut.grad()

    assert grad_ut.is_diag is True
    assert grad_ut.data.shape == (3,)
    assert grad_ut is not ut


def test_diag_permute_dup_index():
    b1 = Bond(dim=3, bond_type=BondType.IN)
    b2 = Bond(dim=3, bond_type=BondType.OUT)

    ut = UniTensor(labels=["a", "b"], bonds=[b1, b2], dtype=float, is_diag=True)

    with pytest.raises(ValueError):
        ut.permute(0, 0)


def test_diag_permute_invalid_index():
    b1 = Bond(dim=3, bond_type=BondType.IN)
    b2 = Bond(dim=3, bond_type=BondType.OUT)

    ut = UniTensor(labels=["a", "b"], bonds=[b1, b2], dtype=float, is_diag=True)

    with pytest.raises(ValueError):
        ut.permute(0, 2)

    with pytest.raises(ValueError):
        ut.permute(-1, 0)


def test_diag_permute():
    b1 = Bond(dim=3, bond_type=BondType.IN)
    b2 = Bond(dim=3, bond_type=BondType.OUT)

    ut = UniTensor(labels=["a", "b"], bonds=[b1, b2], dtype=float, is_diag=True)

    ut2 = ut.permute("b", "a")

    assert ut2.labels == ["b", "a"]
    assert ut2.data is ut.data
