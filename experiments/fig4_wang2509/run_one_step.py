import argparse
import csv
import os

import torch

from experiments.fig4_wang2509.config import DEFAULT_CONFIG, make_prob_vector
from experiments.fig4_wang2509.embeddings import make_embeddings
from experiments.fig4_wang2509.model import init_W, loss_and_grad
from experiments.fig4_wang2509.optimizers import get_direction


def eval_from_direction_logits(ZD: torch.Tensor, eta: float, p: torch.Tensor):
    """
    One-step W = eta * D.
    Instead of recomputing Et.T @ W @ E for every eta,
    precompute ZD = Et.T @ D @ E and use Z = eta * ZD.
    """
    Z = eta * ZD
    logP = torch.log_softmax(Z, dim=0)
    P = torch.exp(logP)

    correct_log_probs = torch.diag(logP)
    correct_probs = torch.diag(P)

    loss = -(p * correct_log_probs).sum()
    delta = correct_probs.max() - correct_probs.min()

    return loss.item(), delta.item(), correct_probs.min().item(), correct_probs.max().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="experiments/fig4_wang2509/results/one_step.csv")
    parser.add_argument("--eta-min", type=float, default=1e-4)
    parser.add_argument("--eta-max", type=float, default=1e5)
    parser.add_argument("--num-etas", type=int, default=300)
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    torch.manual_seed(cfg.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    p = make_prob_vector(cfg)
    eta_grid = torch.logspace(
        torch.log10(torch.tensor(args.eta_min, dtype=torch.float64)),
        torch.log10(torch.tensor(args.eta_max, dtype=torch.float64)),
        args.num_etas,
        dtype=torch.float64,
    ).tolist()

    rows = []

    for regime in ["decoupled", "coupled"]:
        print(f"[one-step] regime={regime}", flush=True)

        E, Et = make_embeddings(regime, cfg)
        W0 = init_W(cfg)

        loss0, grad0, delta0, correct0 = loss_and_grad(W0, E, Et, p)
        print(f"  loss0={loss0.item():.6f}, delta0={delta0.item():.6e}", flush=True)

        for opt in ["gd", "signgd", "muon"]:
            print(f"  optimizer={opt}", flush=True)

            D = get_direction(opt, grad0, tol=cfg.svd_tol)

            # Precompute logits for direction once.
            ZD = Et.T @ D @ E

            for eta in eta_grid:
                loss, delta, cp_min, cp_max = eval_from_direction_logits(ZD, eta, p)

                rows.append(
                    {
                        "regime": regime,
                        "optimizer": opt,
                        "eta": eta,
                        "loss": loss,
                        "delta": delta,
                        "correct_prob_min": cp_min,
                        "correct_prob_max": cp_max,
                        "K": cfg.K,
                        "d": cfg.d,
                        "alpha": cfg.alpha,
                        "beta": cfg.beta,
                        "L": cfg.L,
                    }
                )

    fieldnames = [
        "regime",
        "optimizer",
        "eta",
        "loss",
        "delta",
        "correct_prob_min",
        "correct_prob_max",
        "K",
        "d",
        "alpha",
        "beta",
        "L",
    ]

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] wrote {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
