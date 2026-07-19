import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from experiments.fig4_wang2509.embeddings import rotation_matrix
from experiments.fig4_wang2509.optimizers import muon_direction
from experiments.fig4_wang2509.run_multistep_eta_sweep import (
    SweepConfig,
    coupled_signgd_logit_direction,
    make_prob_vector,
)


OPTIMIZERS = ("gd", "signgd", "muon")
REGIMES = ("decoupled", "coupled")


def metrics_and_gradient(Z, p):
    logP = torch.log_softmax(Z, dim=0)
    P = torch.exp(logP)
    correct_log = torch.diag(logP)
    correct = torch.diag(P)
    loss = -(p * correct_log).sum()
    delta = correct.max() - correct.min()
    eye = torch.eye(Z.shape[0], dtype=Z.dtype, device=Z.device)
    gradient = (P - eye) * p[None, :]
    return loss, delta, gradient


def direction(optimizer, regime, gradient, rotations):
    if optimizer == "gd":
        return -gradient
    if optimizer == "muon":
        return muon_direction(gradient)
    if optimizer == "signgd" and regime == "decoupled":
        return -torch.sign(gradient)
    if optimizer == "signgd" and regime == "coupled":
        return coupled_signgd_logit_direction(gradient, *rotations)
    raise ValueError((optimizer, regime))


def cosine(a, b):
    a_norm = torch.linalg.vector_norm(a)
    b_norm = torch.linalg.vector_norm(b)
    if a_norm.item() == 0.0 or b_norm.item() == 0.0:
        return float("nan")
    return float(torch.sum(a * b).div(a_norm * b_norm).item())


def run_case(optimizer, regime, eta, steps, cfg, p, rotations):
    zero = torch.zeros((cfg.K, cfg.K), dtype=cfg.dtype, device=cfg.device)
    _, _, grad0 = metrics_and_gradient(zero, p)
    direction0 = direction(optimizer, regime, grad0, rotations)

    rows = []
    for protocol in ("iterative", "frozen"):
        Z = zero.clone()
        for step in range(steps + 1):
            loss, delta, gradient = metrics_and_gradient(Z, p)
            if protocol == "frozen":
                current_direction = direction0
                alignment = 1.0
            elif torch.count_nonzero(gradient).item() == 0:
                current_direction = torch.zeros_like(direction0)
                alignment = float("nan")
            else:
                current_direction = direction(optimizer, regime, gradient, rotations)
                alignment = cosine(current_direction, direction0)
            rows.append(
                {
                    "optimizer": optimizer,
                    "regime": regime,
                    "protocol": protocol,
                    "eta": eta,
                    "step": step,
                    "loss": float(loss.item()),
                    "delta": float(delta.item()),
                    "cosine_to_initial": alignment,
                }
            )
            if step == steps:
                break
            update = current_direction if protocol == "iterative" else direction0
            Z = Z + eta * update
    return rows


def plot_metric(df, metric, ylabel, out_path, floor=None):
    colors = {"iterative": "#2166ac", "frozen": "#d73027"}
    styles = {"iterative": "-", "frozen": "--"}
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex=True, sharey=True)

    for row, regime in enumerate(REGIMES):
        for col, optimizer in enumerate(OPTIMIZERS):
            ax = axes[row, col]
            panel = df[(df["optimizer"] == optimizer) & (df["regime"] == regime)]
            for protocol in ("iterative", "frozen"):
                sub = panel[panel["protocol"] == protocol].sort_values("step")
                values = sub[metric].clip(lower=floor) if floor else sub[metric]
                ax.plot(
                    sub["step"],
                    values,
                    label=protocol.capitalize(),
                    color=colors[protocol],
                    linestyle=styles[protocol],
                    linewidth=2.1,
                )
            ax.set_title(f"{optimizer.upper()} × {regime}")
            ax.grid(True, which="both", alpha=0.25)
            if floor:
                ax.set_yscale("log")
            if row == 1:
                ax.set_xlabel("Step")
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0 and col == 0:
                ax.legend(framealpha=0.95)

    fig.suptitle("K=999: iterative vs frozen initial direction", y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out_path}")


def plot_cosine(df, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex=True, sharey=True)
    for row, regime in enumerate(REGIMES):
        for col, optimizer in enumerate(OPTIMIZERS):
            ax = axes[row, col]
            sub = df[
                (df["optimizer"] == optimizer)
                & (df["regime"] == regime)
                & (df["protocol"] == "iterative")
            ].sort_values("step")
            # Once loss reaches the float64 floor, the remaining gradient is
            # zero or numerical residue and its direction is not meaningful.
            sub = sub[sub["loss"] > 1e-16]
            ax.plot(
                sub["step"],
                sub["cosine_to_initial"],
                color="#762a83",
                linewidth=2.0,
            )
            ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
            ax.set_title(f"{optimizer.upper()} × {regime}")
            ax.grid(True, alpha=0.25)
            ax.set_ylim(-1.05, 1.05)
            if row == 1:
                ax.set_xlabel("Step")
            if col == 0:
                ax.set_ylabel(r"cos$(D_t,D_0)$")
    fig.suptitle(
        "K=999: direction alignment before numerical convergence", y=1.01
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=999)
    parser.add_argument("--L", type=int, default=200)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eta-gd", type=float, default=194149.19457438815)
    parser.add_argument("--eta-signgd", type=float, default=0.13)
    parser.add_argument("--eta-muon", type=float, default=22.695105366946684)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = SweepConfig(K=args.K, L=args.L, device="cpu")
    p = make_prob_vector(cfg)
    rotations = (
        rotation_matrix(3.638, 2.949, 5.218, device=cfg.device, dtype=cfg.dtype),
        rotation_matrix(1.715, 0.876, 3.098, device=cfg.device, dtype=cfg.dtype),
    )
    etas = {"gd": args.eta_gd, "signgd": args.eta_signgd, "muon": args.eta_muon}

    rows = []
    for optimizer in OPTIMIZERS:
        for regime in REGIMES:
            print(f"[diagnostic] {optimizer} x {regime}", flush=True)
            rows.extend(
                run_case(
                    optimizer,
                    regime,
                    etas[optimizer],
                    args.steps,
                    cfg,
                    p,
                    rotations,
                )
            )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[done] wrote {len(df)} rows to {args.out_csv}")

    plot_metric(
        df,
        "loss",
        "Population loss",
        os.path.join(args.out_dir, "frozen_vs_iterative_loss.png"),
        floor=1e-16,
    )
    plot_metric(
        df,
        "delta",
        r"$\Delta(W)$",
        os.path.join(args.out_dir, "frozen_vs_iterative_delta.png"),
        floor=1e-16,
    )
    plot_cosine(df, os.path.join(args.out_dir, "direction_cosine_to_initial.png"))


if __name__ == "__main__":
    main()
