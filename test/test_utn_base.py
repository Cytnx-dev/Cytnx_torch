from cytnx_torch.bond import Bond, BondType
from cytnx_torch.unitensor import UniTensor, RegularUniTensor
import numpy as np
import torch


def test_permute():

    b1 = Bond(dim=10, bond_type=BondType.IN)
    b2 = Bond(dim=10, bond_type=BondType.OUT)
    b3 = Bond(dim=20, bond_type=BondType.OUT)

    ut = UniTensor(labels=["a", "b", "c"], bonds=[b1, b2, b3], dtype=float)

    ut_new = ut.permute(1, 0, 2, by_label=False)

    assert np.all(ut_new.labels == ["b", "a", "c"])

    ut_new = ut.permute("c", "a", "b", by_label=True)

    assert np.all(ut_new.labels == ["c", "a", "b"])


def test_init():

    body = torch.zeros(size=[2, 5, 3, 4])

    ut = UniTensor.from_torch(tensor=body, labels=["a", "b", "c", "d"])

    assert isinstance(ut, RegularUniTensor)
    assert isinstance(ut.data, torch.Tensor)
    assert ut.data is body
