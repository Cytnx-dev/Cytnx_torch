import numpy as np
from cytnx_torch.symmetry import U1, Zn


def test_U1():

    s1 = U1(label="a")

    q1 = [0, 1, 2]
    q2 = [0, -1, 3]

    assert s1.check_qnums(q1)
    assert s1.check_qnums(q2)

    assert np.all(s1.combine_qnums(q1, q2) == [0, -1, 3, 1, 0, 4, 2, 1, 5])


def test_Zn():

    s1 = Zn(label="x", n=3)

    q1 = [0, 1, 2]
    q2 = [2, 0]

    assert s1.check_qnums(q1)
    assert s1.check_qnums(q2)

    assert np.all(s1.combine_qnums(q1, q2) == [2, 0, 0, 1, 1, 2])
