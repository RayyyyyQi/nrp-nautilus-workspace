from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Fig4Config:
    # Figure 4 setting
    K: int = 999
    d: int = 999

    # alpha = probability mass on head facts
    # beta = fraction of head facts
    alpha: float = 0.8
    beta: float = 0.2

    # beta * K = 199.8, paper does not specify rounding.
    # We use L = 200 as the main reproduction choice.
    L: int = 200

    dtype_name: str = "float64"
    device: str = "cpu"
    seed: int = 0

    eps: float = 1e-12
    svd_tol: float = 1e-12


DEFAULT_CONFIG = Fig4Config()


def get_dtype(dtype_name: str):
    if dtype_name == "float64":
        return torch.float64
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype_name: {dtype_name}")


def make_prob_vector(cfg: Fig4Config, device=None, dtype=None):
    """
    p_k = alpha / L for k <= L
    p_k = (1 - alpha) / (K - L) for k > L
    """
    device = device or cfg.device
    dtype = dtype or get_dtype(cfg.dtype_name)

    p = torch.empty(cfg.K, device=device, dtype=dtype)
    p[: cfg.L] = cfg.alpha / cfg.L
    p[cfg.L :] = (1.0 - cfg.alpha) / (cfg.K - cfg.L)

    return p
