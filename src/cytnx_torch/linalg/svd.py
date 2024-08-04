from cytnx_torch.bond import Bond, BondType  # noqa F401
from cytnx_torch.unitensor.regular_unitensor import RegularUniTensor
from torch import linalg as torch_linalg  # noqa F401


def _svd_regular_tn(
    A: RegularUniTensor, truncate_dim: int = None, truncate_tol: float = None
):
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

    if A.is_diag:
        raise ValueError("TODO SVD is not supported for diagonal tensors for now")

    mat_A, cL, cR = A.as_matrix(left_bond_label="_tmp_L_", right_bond_label="_tmp_R_")

    # get the data:
    u_dat, s_dat, v_dat = torch_linalg.svd(mat_A.data)

    # create new bonds:
    new_bond_dim = len(s_dat)
    internal_bond = Bond(
        new_bond_dim,
        bond_type=BondType.OUT if A.is_directional_bonds else BondType.NONE,
    )

    # construct U tensor:
    u_labels = ["_tmp_L_", "_aux_L_"]
    u_bonds = [mat_A.bonds[0], internal_bond]
    u = RegularUniTensor(labels=u_labels, bonds=u_bonds, is_diag=False, data=u_dat)

    # construct s tensor:
    s_labels = ["_aux_L_", "_aux_R_"]
    s_bonds = [internal_bond.redirect(), internal_bond]
    s = RegularUniTensor(labels=s_labels, bonds=s_bonds, is_diag=True, data=s_dat)

    # construct v tensor:
    v_labels = ["_aux_R_", "_tmp_R_"]
    v_bonds = [internal_bond, mat_A.bonds[1]]
    v = RegularUniTensor(labels=v_labels, bonds=v_bonds, is_diag=False, data=v_dat)

    if truncate_dim or truncate_tol:
        # deal truncate
        # TODO
        pass

    new_mat = u.contract(s).contract(v)

    return cL.contract(new_mat).contract(cR)
