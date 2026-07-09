import math
import torch

from experiments.fig4_wang2509.config import Fig4Config, get_dtype


def init_W(cfg: Fig4Config):
    dtype = get_dtype(cfg.dtype_name)
    return torch.zeros((cfg.d, cfg.K), device=cfg.device, dtype=dtype)


def logits(W: torch.Tensor, E: torch.Tensor, Et: torch.Tensor):
    """
    Z[j, k] = Et[:, j]^T W E[:, k]

    Column k is the query/fact index.
    Row j is the candidate output/object index.
    """
    return Et.T @ W @ E


def probabilities_from_logits(Z: torch.Tensor):
    """
    Softmax over candidate objects j for each fixed query k.
    """
    return torch.softmax(Z, dim=0)


def loss_delta_and_correct_probs(
    W: torch.Tensor,
    E: torch.Tensor,
    Et: torch.Tensor,
    p: torch.Tensor,
    eps: float = 1e-12,
):
    Z = logits(W, E, Et)

    # log_softmax is more stable than log(softmax(.))
    logP = torch.log_softmax(Z, dim=0)
    P = torch.exp(logP)

    correct_log_probs = torch.diag(logP)
    correct_probs = torch.diag(P)

    loss = -(p * correct_log_probs).sum()
    delta = correct_probs.max() - correct_probs.min()

    return loss, delta, correct_probs


def loss_and_grad(
    W: torch.Tensor,
    E: torch.Tensor,
    Et: torch.Tensor,
    p: torch.Tensor,
):
    """
    Population cross-entropy loss and manual gradient.

    Loss:
        L(W) = - sum_k p_k log softmax(Et.T @ W @ E)[k, k]

    Gradient:
        dL/dZ[:, k] = p_k * (P[:, k] - e_k)
        grad_W = Et @ M @ E.T
    """
    Z = logits(W, E, Et)
    logP = torch.log_softmax(Z, dim=0)
    P = torch.exp(logP)

    K = Z.shape[0]
    I = torch.eye(K, device=Z.device, dtype=Z.dtype)

    M = (P - I) * p[None, :]
    grad = Et @ M @ E.T

    correct_log_probs = torch.diag(logP)
    correct_probs = torch.diag(P)

    loss = -(p * correct_log_probs).sum()
    delta = correct_probs.max() - correct_probs.min()

    return loss, grad, delta, correct_probs
