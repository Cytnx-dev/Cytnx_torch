from cytnx_torch.bond import Bond, BondType  # noqa F401
from cytnx_torch.unitensor.regular_unitensor import RegularUniTensor
from torch import linalg as torch_linalg  # noqa F401


def _svd_regular_tn(A: RegularUniTensor):
    """
    svd(A, **kwargs):
        Singular Value Decomposition of a matrix A.
        [U,S,V] = svd(A) returns the singular value decomposition of A such that A = U*S*V^T.
        The input tensor can be complex.

    Args:
        A (cytnx.UniTensor): the input tensor

    Returns:
        U (cytnx.UniTensor): the left singular vectors
        S (cytnx.UniTensor): the singular values
        V (cytnx.UniTensor): the right singular vectors
    """

    mat_A, cL, cR = A.as_matrix(left_bond_label="_tmp_L_", right_bond_label="_tmp_R_")

    # get the data:
    # u,s,v = torch_linalg.svd(mat_A)

    # create new bonds:
    # new_bond_dim = len(s)
    # internal_bond = Bond(new_bond_dim,bond_type=BondType.OUT)
