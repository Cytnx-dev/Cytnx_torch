import torch


def _tensordot_dg(
    A: torch.Tensor,
    B: torch.Tensor,
    is_diag_left: bool = False,
    is_diag_right: bool = False,
):
    """
    Matrix multiplication of two tensors A and B.

    Args:
        A (torch.Tensor): the first tensor
        B (torch.Tensor): the second tensor

    Returns:
        torch.Tensor: the result of the matrix multiplication
    """

    if is_diag_left and is_diag_right:
        return A * B

    if is_diag_left:
        # check shape:
        pass
