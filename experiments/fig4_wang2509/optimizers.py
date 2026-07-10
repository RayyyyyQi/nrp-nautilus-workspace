import torch


def gd_direction(grad):
    return -grad


def signgd_direction(grad):
    return -torch.sign(grad)


def muon_direction(grad, tol=1e-12):
    """
    Exact Muon / polar direction for -grad.

    Primary implementation:
        G = U S V^T
        polar(G) = U V^T on nonzero singular directions

    Fallback:
        polar(G) = G (G^T G)^(-1/2)

    The fallback is mathematically the same polar-map direction, but more
    robust when torch.linalg.svd fails on ill-conditioned matrices.
    """
    G = -grad

    try:
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        mask = S > tol

        if mask.sum().item() == 0:
            return torch.zeros_like(G)

        return U[:, mask] @ Vh[mask, :]

    except torch._C._LinAlgError:
        A = G.T @ G
        A = 0.5 * (A + A.T)

        evals, V = torch.linalg.eigh(A)
        evals = torch.clamp(evals, min=0.0)

        mask = evals > (tol ** 2)

        if mask.sum().item() == 0:
            return torch.zeros_like(G)

        V_keep = V[:, mask]
        inv_sqrt = torch.rsqrt(evals[mask])

        return (G @ V_keep) @ (inv_sqrt[:, None] * V_keep.T)


def get_direction(name, grad, tol=1e-12):
    name = name.lower()

    if name == "gd":
        return gd_direction(grad)

    if name == "signgd":
        return signgd_direction(grad)

    if name == "muon":
        return muon_direction(grad, tol=tol)

    raise ValueError(f"Unknown optimizer: {name}")
