import argparse
import csv
import os
from dataclasses import replace

import torch

from .config import DEFAULT_CONFIG, get_dtype, make_prob_vector
from .embeddings import make_embeddings
from .model import init_W, logits, probabilities_from_logits
from .optimizers import get_direction


def loss_delta_grad_correct(W, E, Et, p, cfg):
    """
    Compute population loss, Delta(W), gradient, and correct probabilities.

    Z[j, k] = Et[:, j]^T W E[:, k]
    column k = query/fact index
    row j = candidate output index
    softmax is over rows, i.e. dim=0
    """
    Z = logits(W, E, Et)
    logP = torch.log_softmax(Z, dim=0)
    P = torch.exp(logP)

    correct_probs = torch.diag(P)

    loss = -(p * torch.diag(logP)).sum()
    delta = correct_probs.max() - correct_probs.min()

    I = torch.eye(cfg.K, device=W.device, dtype=W.dtype)
    M = (P - I) * p[None, :]
    grad = Et @ M @ E.T

    return loss, delta, grad, correct_probs


def run_one_trajectory(regime, optimizer, eta, steps, stop_loss, cfg):
    dtype = get_dtype(cfg.dtype_name)
    device = cfg.device

    p = make_prob_vector(cfg, device=device, dtype=dtype)
    E, Et = make_embeddings(regime, cfg)
    W = init_W(cfg)

    rows = []

    for step in range(steps + 1):
        loss, delta, grad, correct_probs = loss_delta_grad_correct(W, E, Et, p, cfg)

        rows.append(
            {
                "regime": regime,
                "optimizer": optimizer,
                "step": step,
                "eta": eta,
                "loss": float(loss.item()),
                "delta": float(delta.item()),
                "correct_min": float(correct_probs.min().item()),
                "correct_max": float(correct_probs.max().item()),
            }
        )

        if step == steps:
            break

        if float(loss.item()) <= stop_loss:
            break

        D = get_direction(optimizer, grad, tol=cfg.svd_tol)
        W = W + eta * D

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", type=str, default="experiments/fig4_wang2509/results/multi_step.csv")

    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--stop-loss", type=float, default=5e-2)

    parser.add_argument("--L", type=int, default=DEFAULT_CONFIG.L)
    parser.add_argument("--K", type=int, default=DEFAULT_CONFIG.K)
    parser.add_argument("--device", default=DEFAULT_CONFIG.device)

    # Baseline learning rates. These are reproduction choices and can be tuned.
    parser.add_argument("--eta-gd", type=float, default=2000.0)
    parser.add_argument("--eta-signgd", type=float, default=0.01)
    parser.add_argument("--eta-muon", type=float, default=1.0)

    args = parser.parse_args()

    cfg = replace(DEFAULT_CONFIG, K=args.K, d=args.K, L=args.L, device=args.device)

    eta_by_optimizer = {
        "gd": args.eta_gd,
        "signgd": args.eta_signgd,
        "muon": args.eta_muon,
    }

    all_rows = []

    for regime in ["decoupled", "coupled"]:
        print(f"[multi-step] regime={regime}")

        for optimizer in ["gd", "signgd", "muon"]:
            eta = eta_by_optimizer[optimizer]
            print(f"  optimizer={optimizer}, eta={eta}")

            rows = run_one_trajectory(
                regime=regime,
                optimizer=optimizer,
                eta=eta,
                steps=args.steps,
                stop_loss=args.stop_loss,
                cfg=cfg,
            )

            last = rows[-1]
            print(
                f"    final_step={last['step']}, "
                f"final_loss={last['loss']:.6e}, "
                f"final_delta={last['delta']:.6e}"
            )

            all_rows.extend(rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fieldnames = [
        "regime",
        "optimizer",
        "step",
        "eta",
        "loss",
        "delta",
        "correct_min",
        "correct_max",
    ]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[done] wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
