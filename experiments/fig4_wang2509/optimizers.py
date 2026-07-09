import torch


def gd_direction(grad: torch.Tensor):
    """
    Standard gradient descent direction.
    W <- W + eta * D, where D = -grad.
    """
    return -grad


def signgd_direction(grad: torch.Tensor):
    """
    SignGD / simplified Adam-like coordinate-wise normalized direction.
    W <- W + eta * D, where D = -sign(grad).
    """
    return -torch.sign(grad)


def muon_direction(grad: torch.Tensor, tol: float = 1e-12):
    """
    Exact-SVD Muon direction.

    We apply polar map to the negative gradient:
        G = -grad = U S V^T
        D = U V^T over nonzero singular directions

    Important:
        Do not blindly use U @ Vh if rank-deficient.
        We mask tiny singular values.
    """
    G = -grad
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)

    mask = S > tol
    if mask.sum().item() == 0:
        return torch.zeros_like(G)

    return U[:, mask] @ Vh[mask, :]


def get_direction(name: str, grad: torch.Tensor, tol: float = 1e-12):
    name = name.lower()

    if name == "gd":
        return gd_direction(grad)

    if name == "signgd":
        return signgd_direction(grad)

    if name == "muon":
        return muon_direction(grad, tol=tol)

    raise ValueError(f"Unknown optimizer: {name}")
