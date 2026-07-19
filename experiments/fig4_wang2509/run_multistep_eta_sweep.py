import argparse
import os
from dataclasses import dataclass

import pandas as pd
import torch

from experiments.fig4_wang2509.embeddings import rotation_matrix
from experiments.fig4_wang2509.optimizers import muon_direction


@dataclass(frozen=True)
class SweepConfig:
    K: int = 300
    L: int = 60
    alpha: float = 0.8
    dtype: torch.dtype = torch.float64
    device: str = "cpu"


def make_prob_vector(cfg):
    p = torch.empty(cfg.K, dtype=cfg.dtype, device=cfg.device)
    p[: cfg.L] = cfg.alpha / cfg.L
    p[cfg.L :] = (1.0 - cfg.alpha) / (cfg.K - cfg.L)
    return p


def loss_delta_and_grad_z(Z, p):
    logP = torch.log_softmax(Z, dim=0)
    P = torch.exp(logP)
    correct_log = torch.diag(logP)
    correct = torch.diag(P)
    loss = -(p * correct_log).sum()
    delta = correct.max() - correct.min()

    I = torch.eye(Z.shape[0], dtype=Z.dtype, device=Z.device)
    M = (P - I) * p[None, :]
    return loss, delta, M


def coupled_signgd_logit_direction(M, R_tilde, R_e):
    """Apply sign in ambient W coordinates using independent 3x3 blocks."""
    num_blocks = M.shape[0] // 3
    blocks = M.reshape(num_blocks, 3, num_blocks, 3)

    # grad_W[a,r,b,s] = Rt[r,i] M[a,i,b,j] Re[s,j]
    grad_w = torch.einsum("ri,aibj,sj->arbs", R_tilde, blocks, R_e)
    direction_w = -torch.sign(grad_w)

    # D_Z[a,i,b,j] = Rt[r,i] D_W[a,r,b,s] Re[s,j]
    direction_z = torch.einsum("ri,arbs,sj->aibj", R_tilde, direction_w, R_e)
    return direction_z.reshape_as(M)


def run_trajectory(optimizer, regime, eta, checkpoints, cfg, p, rotations):
    Z = torch.zeros((cfg.K, cfg.K), dtype=cfg.dtype, device=cfg.device)
    checkpoint_set = set(checkpoints)
    rows = []

    for step in range(1, max(checkpoints) + 1):
        _, _, M = loss_delta_and_grad_z(Z, p)

        if optimizer == "gd":
            direction_z = -M
        elif optimizer == "muon":
            direction_z = muon_direction(M)
        elif optimizer == "signgd" and regime == "decoupled":
            direction_z = -torch.sign(M)
        elif optimizer == "signgd" and regime == "coupled":
            direction_z = coupled_signgd_logit_direction(M, *rotations)
        else:
            raise ValueError((optimizer, regime))

        Z = Z + eta * direction_z

        if step in checkpoint_set:
            loss, delta, _ = loss_delta_and_grad_z(Z, p)
            rows.append(
                {
                    "optimizer": optimizer,
                    "regime": regime,
                    "eta": eta,
                    "steps": step,
                    "loss": float(loss.item()),
                    "delta": float(delta.item()),
                }
            )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=300)
    parser.add_argument("--L", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-etas", type=int, default=60)
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 50, 100, 200, 500])
    parser.add_argument("--eta-gd", type=float, nargs=2, default=[1e-1, 1e4])
    parser.add_argument("--eta-signgd", type=float, nargs=2, default=[1e-4, 1e1])
    parser.add_argument("--eta-muon", type=float, nargs=2, default=[1e-4, 1e1])
    parser.add_argument(
        "--out",
        default="experiments/fig4_wang2509/results/multistep_eta_sweep_K300.csv",
    )
    args = parser.parse_args()

    if args.K % 3 != 0:
        raise ValueError("K must be divisible by 3 for the coupled construction")

    cfg = SweepConfig(K=args.K, L=args.L, device=args.device)
    p = make_prob_vector(cfg)
    rotations = (
        rotation_matrix(3.638, 2.949, 5.218, device=cfg.device, dtype=cfg.dtype),
        rotation_matrix(1.715, 0.876, 3.098, device=cfg.device, dtype=cfg.dtype),
    )
    eta_ranges = {
        "gd": args.eta_gd,
        "signgd": args.eta_signgd,
        "muon": args.eta_muon,
    }

    rows = []
    for optimizer in ("gd", "signgd", "muon"):
        lo, hi = eta_ranges[optimizer]
        etas = torch.logspace(
            torch.log10(torch.tensor(lo, dtype=cfg.dtype)),
            torch.log10(torch.tensor(hi, dtype=cfg.dtype)),
            args.num_etas,
            dtype=cfg.dtype,
        ).tolist()

        for regime in ("decoupled", "coupled"):
            # GD and Muon are representation invariant here. Compute once and
            # duplicate the rows for the coupled panel.
            if optimizer in ("gd", "muon") and regime == "coupled":
                copied = [
                    {**row, "regime": "coupled"}
                    for row in rows
                    if row["optimizer"] == optimizer and row["regime"] == "decoupled"
                ]
                rows.extend(copied)
                continue

            print(f"[{optimizer} x {regime}] {len(etas)} eta values", flush=True)
            for index, eta in enumerate(etas, start=1):
                rows.extend(
                    run_trajectory(
                        optimizer, regime, eta, args.steps, cfg, p, rotations
                    )
                )
                if index % 5 == 0 or index == len(etas):
                    print(f"  completed {index}/{len(etas)}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[done] wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
