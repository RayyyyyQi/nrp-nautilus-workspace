import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import torch


@dataclass
class AblationConfig:
    K: int = 300
    L: int = 60
    block_size: int = 3
    dtype: torch.dtype = torch.float64
    device: str = "cpu"


def make_prob_vector(cfg: AblationConfig):
    K, L = cfg.K, cfg.L
    alpha = 0.8

    p = torch.empty(K, dtype=cfg.dtype, device=cfg.device)
    p[:L] = alpha / L
    p[L:] = (1.0 - alpha) / (K - L)
    return p


def make_identity_decoupled(cfg: AblationConfig):
    K = cfg.K
    E = torch.eye(K, dtype=cfg.dtype, device=cfg.device)
    Et = torch.eye(K, dtype=cfg.dtype, device=cfg.device)
    return E, Et


def make_equal_block_decoupled(cfg: AblationConfig):
    """
    Disjoint-support embeddings with identical coordinate geometry.

    Each fact owns a block of size b.
    All blocks use the same normalized pattern.

    Expected behavior:
      pure SignGD should still be almost perfectly balanced.
    """
    K, b = cfg.K, cfg.block_size
    d = K * b

    pattern = torch.tensor([1.0, -0.5, 0.25], dtype=cfg.dtype, device=cfg.device)
    pattern = pattern[:b]
    pattern = pattern / torch.linalg.norm(pattern)

    E = torch.zeros(d, K, dtype=cfg.dtype, device=cfg.device)
    Et = torch.zeros(d, K, dtype=cfg.dtype, device=cfg.device)

    for k in range(K):
        sl = slice(k * b, (k + 1) * b)
        E[sl, k] = pattern
        Et[sl, k] = pattern

    return E, Et


def make_hetero_block_decoupled(cfg: AblationConfig):
    """
    Disjoint-support embeddings with heterogeneous coordinate geometry.

    Each fact still owns its own block, so supports are disjoint.
    But different facts use different coordinate patterns.

    Expected behavior:
      pure SignGD can become imbalanced because sign(grad) is coordinate-wise
      and depends on the L1 geometry of the embedding coordinates.
    """
    K, b = cfg.K, cfg.block_size
    d = K * b

    base_patterns = [
        torch.tensor([1.0, 0.0, 0.0], dtype=cfg.dtype, device=cfg.device),
        torch.tensor([1.0, 1.0, 0.0], dtype=cfg.dtype, device=cfg.device),
        torch.tensor([1.0, 1.0, 1.0], dtype=cfg.dtype, device=cfg.device),
    ]

    base_patterns = [v[:b] / torch.linalg.norm(v[:b]) for v in base_patterns]

    E = torch.zeros(d, K, dtype=cfg.dtype, device=cfg.device)
    Et = torch.zeros(d, K, dtype=cfg.dtype, device=cfg.device)

    for k in range(K):
        sl = slice(k * b, (k + 1) * b)

        # Cycle through different L1 geometries.
        pat_e = base_patterns[k % len(base_patterns)]
        pat_t = base_patterns[(k + 1) % len(base_patterns)]

        E[sl, k] = pat_e
        Et[sl, k] = pat_t

    return E, Et


def make_embeddings(kind: str, cfg: AblationConfig):
    if kind == "identity_decoupled":
        return make_identity_decoupled(cfg)

    if kind == "equal_block_decoupled":
        return make_equal_block_decoupled(cfg)

    if kind == "hetero_block_decoupled":
        return make_hetero_block_decoupled(cfg)

    raise ValueError(f"Unknown embedding kind: {kind}")


def logits(W, E, Et):
    return Et.T @ W @ E


def loss_delta_correct(W, E, Et, p):
    Z = logits(W, E, Et)
    P = torch.softmax(Z, dim=0)
    correct = torch.diag(P)

    loss = -(p * torch.log(correct)).sum()
    delta = correct.max() - correct.min()

    return loss.item(), delta.item(), correct.min().item(), correct.max().item()


def grad_loss(W, E, Et, p):
    K = E.shape[1]

    Z = logits(W, E, Et)
    P = torch.softmax(Z, dim=0)

    I = torch.eye(K, dtype=W.dtype, device=W.device)
    M = (P - I) * p[None, :]

    grad = Et @ M @ E.T
    return grad


def run_one_step(kind, cfg, eta_min, eta_max, num_etas):
    E, Et = make_embeddings(kind, cfg)
    p = make_prob_vector(cfg)

    W0 = torch.zeros(Et.shape[0], E.shape[0], dtype=cfg.dtype, device=cfg.device)
    grad0 = grad_loss(W0, E, Et, p)
    D = -torch.sign(grad0)

    etas = torch.logspace(
        torch.log10(torch.tensor(eta_min, dtype=cfg.dtype)),
        torch.log10(torch.tensor(eta_max, dtype=cfg.dtype)),
        num_etas,
        dtype=cfg.dtype,
        device=cfg.device,
    )

    rows = []
    for eta in etas:
        W = eta * D
        loss, delta, cmin, cmax = loss_delta_correct(W, E, Et, p)
        rows.append(
            {
                "kind": kind,
                "protocol": "one_step",
                "eta": float(eta.item()),
                "step": 1,
                "loss": loss,
                "delta": delta,
                "correct_min": cmin,
                "correct_max": cmax,
            }
        )

    return rows


def run_multi_step(kind, cfg, eta, steps, stop_loss):
    E, Et = make_embeddings(kind, cfg)
    p = make_prob_vector(cfg)

    W = torch.zeros(Et.shape[0], E.shape[0], dtype=cfg.dtype, device=cfg.device)

    rows = []

    for step in range(steps + 1):
        loss, delta, cmin, cmax = loss_delta_correct(W, E, Et, p)
        rows.append(
            {
                "kind": kind,
                "protocol": "multi_step",
                "eta": eta,
                "step": step,
                "loss": loss,
                "delta": delta,
                "correct_min": cmin,
                "correct_max": cmax,
            }
        )

        if step >= 1 and loss <= stop_loss:
            break

        grad = grad_loss(W, E, Et, p)
        D = -torch.sign(grad)
        W = W + eta * D

    return rows


def plot_one_step(df, out_path):
    plt.figure(figsize=(7.5, 5.2))

    styles = {
        "identity_decoupled": ("Identity decoupled", "-"),
        "equal_block_decoupled": ("Equal-block decoupled", "--"),
        "hetero_block_decoupled": ("Heterogeneous-block decoupled", "-."),
    }

    for kind, (label, linestyle) in styles.items():
        sub = df[(df["kind"] == kind) & (df["protocol"] == "one_step")].copy()
        sub = sub.sort_values("loss")

        sub = sub[(sub["loss"] >= 5e-2) & (sub["loss"] <= 1.5e1)].copy()
        sub["delta_plot"] = sub["delta"].clip(lower=1e-7, upper=1.0)

        plt.plot(
            sub["loss"],
            sub["delta_plot"],
            label=label,
            linestyle=linestyle,
            linewidth=2.0,
        )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlim(5e-2, 1.5e1)
    plt.ylim(8e-8, 1.5e0)

    plt.xticks(
        [1e-1, 1e0, 1e1],
        [r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"],
    )

    plt.yticks(
        [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        ["1e+0", "1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6", "0"],
    )

    plt.xlabel("Population Loss")
    plt.ylabel(r"$\Delta(W)$")
    plt.title("SignGD decoupled ablation: one-step")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250)
    print(f"[done] wrote {out_path}")


def plot_multi_step(df, out_path):
    plt.figure(figsize=(7.5, 5.2))

    styles = {
        "identity_decoupled": ("Identity decoupled", "-"),
        "equal_block_decoupled": ("Equal-block decoupled", "--"),
        "hetero_block_decoupled": ("Heterogeneous-block decoupled", "-."),
    }

    for kind, (label, linestyle) in styles.items():
        sub = df[(df["kind"] == kind) & (df["protocol"] == "multi_step")].copy()
        sub = sub.sort_values("step")

        sub = sub[(sub["loss"] >= 5e-2) & (sub["loss"] <= 1.5e1)].copy()
        sub["delta_plot"] = sub["delta"].clip(lower=1e-7, upper=1.0)

        plt.plot(
            sub["loss"],
            sub["delta_plot"],
            label=label,
            linestyle=linestyle,
            linewidth=2.0,
            marker="o",
            markersize=3,
            markevery=max(1, len(sub) // 12),
        )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlim(5e-2, 1.5e1)
    plt.ylim(8e-8, 1.5e0)

    plt.xticks(
        [1e-1, 1e0, 1e1],
        [r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"],
    )

    plt.yticks(
        [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        ["1e+0", "1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6", "0"],
    )

    plt.xlabel("Population Loss")
    plt.ylabel(r"$\Delta(W)$")
    plt.title("SignGD decoupled ablation: multi-step")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250)
    print(f"[done] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=300)
    parser.add_argument("--L", type=int, default=60)
    parser.add_argument("--block-size", type=int, default=3)

    parser.add_argument("--eta-min", type=float, default=1e-3)
    parser.add_argument("--eta-max", type=float, default=1e2)
    parser.add_argument("--num-etas", type=int, default=160)

    parser.add_argument("--multi-eta", type=float, default=0.15)
    parser.add_argument("--multi-steps", type=int, default=80)
    parser.add_argument("--stop-loss", type=float, default=2e-2)

    parser.add_argument(
        "--out-csv",
        type=str,
        default="experiments/fig4_wang2509/results/signgd_decoupled_ablation.csv",
    )
    parser.add_argument(
        "--out-one-step-fig",
        type=str,
        default="experiments/fig4_wang2509/figures/signgd_decoupled_ablation_one_step.png",
    )
    parser.add_argument(
        "--out-multi-step-fig",
        type=str,
        default="experiments/fig4_wang2509/figures/signgd_decoupled_ablation_multi_step.png",
    )

    args = parser.parse_args()

    cfg = AblationConfig(K=args.K, L=args.L, block_size=args.block_size)

    kinds = [
        "identity_decoupled",
        "equal_block_decoupled",
        "hetero_block_decoupled",
    ]

    rows = []

    for kind in kinds:
        print("=" * 80)
        print(f"[one-step] {kind}")
        rows.extend(
            run_one_step(
                kind=kind,
                cfg=cfg,
                eta_min=args.eta_min,
                eta_max=args.eta_max,
                num_etas=args.num_etas,
            )
        )

        print(f"[multi-step] {kind}")
        rows.extend(
            run_multi_step(
                kind=kind,
                cfg=cfg,
                eta=args.multi_eta,
                steps=args.multi_steps,
                stop_loss=args.stop_loss,
            )
        )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[done] wrote {args.out_csv}")

    plot_one_step(df, args.out_one_step_fig)
    plot_multi_step(df, args.out_multi_step_fig)

    print()
    print("Summary: max delta by kind/protocol")
    summary = (
        df.groupby(["kind", "protocol"])["delta"]
        .max()
        .reset_index()
        .sort_values(["protocol", "kind"])
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
