import math
import torch

from experiments.fig4_wang2509.config import Fig4Config, get_dtype


def rotation_matrix(a: float, b: float, c: float, *, device: str, dtype: torch.dtype):
    """
    R(a,b,c) from the paper's Appendix D construction.
    It should satisfy R.T @ R = I_3.
    """
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cc, sc = math.cos(c), math.sin(c)

    R = torch.tensor(
        [
            [ca * cb * cc - sa * sc, -ca * cb * sc - sa * cc, ca * sb],
            [sa * cb * cc + ca * sc, -sa * cb * sc + ca * cc, sa * sb],
            [-sb * cc, sb * sc, cb],
        ],
        device=device,
        dtype=dtype,
    )
    return R


def make_embeddings(regime: str, cfg: Fig4Config):
    """
    Returns:
        E:  subject-relation embeddings, shape K x K
        Et: object embeddings, shape K x K

    Convention:
        logits Z = Et.T @ W @ E
        column k corresponds to query E[:, k]
    """
    dtype = get_dtype(cfg.dtype_name)
    device = cfg.device
    K = cfg.K

    if cfg.d != cfg.K:
        raise ValueError("This Figure 4 reproduction assumes d = K.")

    if regime == "decoupled":
        E = torch.eye(K, device=device, dtype=dtype)
        Et = torch.eye(K, device=device, dtype=dtype)
        return E, Et

    if regime == "coupled":
        if K % 3 != 0:
            raise ValueError("Coupled construction requires K % 3 == 0.")

        num_blocks = K // 3
        I_blocks = torch.eye(num_blocks, device=device, dtype=dtype)

        R_tilde = rotation_matrix(3.638, 2.949, 5.218, device=device, dtype=dtype)
        R_E = rotation_matrix(1.715, 0.876, 3.098, device=device, dtype=dtype)

        Et = torch.kron(I_blocks, R_tilde).contiguous()
        E = torch.kron(I_blocks, R_E).contiguous()
        return E, Et

    raise ValueError(f"Unknown regime: {regime}. Use 'decoupled' or 'coupled'.")


def orthogonality_error(A: torch.Tensor):
    K = A.shape[1]
    I = torch.eye(K, device=A.device, dtype=A.dtype)
    return torch.max(torch.abs(A.T @ A - I)).item()
