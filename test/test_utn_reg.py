from cytnx_torch.bond import Bond, BondType
from cytnx_torch.unitensor import UniTensor, RegularUniTensor

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
