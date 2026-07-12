import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import torch

from experiments.fig4_wang2509.optimizers import muon_direction


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
    Support-decoupled tall matrix.

    Each fact has its own disjoint block.
    Every block uses the same coordinate pattern.
    """
    K, b = cfg.K, cfg.block_size
    d = K * b

    # Same local pattern for every fact.
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
    Support-decoupled tall matrix.

    Each fact still has disjoint global support.
    But the local coordinate geometry differs across blocks.

    This is a stress test for coordinate-wise SignGD.
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

        # E and Etilde use slightly shifted local patterns.
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


def logits_from_W(W, E, Et):
    return Et.T @ W @ E


def loss_delta_from_Z(Z, p):
    P = torch.softmax(Z, dim=0)
    correct = torch.diag(P)

    loss = -(p * torch.log(correct)).sum()
    delta = correct.max() - correct.min()

    return loss.item(), delta.item(), correct.min().item(), correct.max().item(), P


def grad_Z_from_P(P, p):
    K = P.shape[0]
    I = torch.eye(K, dtype=P.dtype, device=P.device)

    # dL/dZ, column k weighted by p_k.
    M = (P - I) * p[None, :]
    return M


def run_gd_or_muon(kind, optimizer, cfg, eta, steps, stop_loss):
    """
    For GD and Muon, if E and Etilde have orthonormal columns,
    the induced dynamics in logit space Z are independent of
    the ambient coordinate geometry.

    GD:
        Z <- Z - eta * M

    Muon:
        Z <- Z + eta * polar(-M)

    This avoids doing large ambient SVDs on W.
    """
    p = make_prob_vector(cfg)

    Z = torch.zeros(cfg.K, cfg.K, dtype=cfg.dtype, device=cfg.device)

    rows = []

    for step in range(steps + 1):
        loss, delta, cmin, cmax, P = loss_delta_from_Z(Z, p)

        rows.append(
            {
                "kind": kind,
                "optimizer": optimizer,
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

        M = grad_Z_from_P(P, p)

        if optimizer == "gd":
            Z = Z - eta * M
        elif optimizer == "muon":
            D = muon_direction(M)
            Z = Z + eta * D
        else:
            raise ValueError(optimizer)

    return rows


def run_signgd(kind, cfg, eta, steps, stop_loss):
    """
    For SignGD, we must run in ambient W space.

    Reason:
      sign(Et @ M @ E.T) is coordinate-wise and depends on
      the ambient coordinate geometry of E and Etilde.
    """
    E, Et = make_embeddings(kind, cfg)
    p = make_prob_vector(cfg)

    d_out = Et.shape[0]
    d_in = E.shape[0]

    W = torch.zeros(d_out, d_in, dtype=cfg.dtype, device=cfg.device)

    rows = []

    for step in range(steps + 1):
        Z = logits_from_W(W, E, Et)
        loss, delta, cmin, cmax, P = loss_delta_from_Z(Z, p)

        rows.append(
            {
                "kind": kind,
                "optimizer": "signgd",
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

        M = grad_Z_from_P(P, p)
        grad_W = Et @ M @ E.T
        D = -torch.sign(grad_W)

        W = W + eta * D

    return rows


def run_all(cfg, eta_gd, eta_signgd, eta_muon, steps, stop_loss):
    kinds = [
        "identity_decoupled",
        "equal_block_decoupled",
        "hetero_block_decoupled",
    ]

    rows = []

    for kind in kinds:
        print("=" * 80)
        print(f"[kind] {kind}")

        # Sanity check: columns are orthonormal.
        E, Et = make_embeddings(kind, cfg)
        err_E = torch.linalg.norm(E.T @ E - torch.eye(cfg.K, dtype=cfg.dtype)).item()
        err_Et = torch.linalg.norm(Et.T @ Et - torch.eye(cfg.K, dtype=cfg.dtype)).item()
        print(f"orthogonality error: E={err_E:.3e}, Etilde={err_Et:.3e}")

        print("[gd]")
        rows.extend(
            run_gd_or_muon(
                kind=kind,
                optimizer="gd",
                cfg=cfg,
                eta=eta_gd,
                steps=steps,
                stop_loss=stop_loss,
            )
        )

        print("[signgd]")
        rows.extend(
            run_signgd(
                kind=kind,
                cfg=cfg,
                eta=eta_signgd,
                steps=steps,
                stop_loss=stop_loss,
            )
        )

        print("[muon]")
        rows.extend(
            run_gd_or_muon(
                kind=kind,
                optimizer="muon",
                cfg=cfg,
                eta=eta_muon,
                steps=steps,
                stop_loss=stop_loss,
            )
        )

    return pd.DataFrame(rows)


def plot_optimizer(df, optimizer, out_path):
    plt.figure(figsize=(7.5, 5.2))

    styles = {
        "identity_decoupled": ("Identity decoupled", "-"),
        "equal_block_decoupled": ("Equal-block decoupled", "--"),
        "hetero_block_decoupled": ("Heterogeneous-block decoupled", "-."),
    }

    for kind, (label, linestyle) in styles.items():
        sub = df[(df["kind"] == kind) & (df["optimizer"] == optimizer)].copy()
        sub = sub.sort_values("step")

        # Keep only visible positive-loss range.
        sub = sub[(sub["loss"] >= 5e-2) & (sub["loss"] <= 1.5e1)].copy()

        if len(sub) == 0:
            continue

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
    plt.title(f"Decoupled embedding ablation: {optimizer}")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250)
    print(f"[done] wrote {out_path}")


def print_summary(df):
    print()
    print("Summary: max delta by kind and optimizer")
    summary = (
        df.groupby(["optimizer", "kind"])["delta"]
        .max()
        .reset_index()
        .sort_values(["optimizer", "kind"])
    )
    print(summary.to_string(index=False))

    print()
    print("Final rows")
    finals = (
        df.sort_values("step")
        .groupby(["optimizer", "kind"])
        .tail(1)
        .sort_values(["optimizer", "kind"])
    )
    print(
        finals[
            ["optimizer", "kind", "step", "loss", "delta", "correct_min", "correct_max"]
        ].to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--K", type=int, default=300)
    parser.add_argument("--L", type=int, default=60)
    parser.add_argument("--block-size", type=int, default=3)

    parser.add_argument("--eta-gd", type=float, default=250.0)
    parser.add_argument("--eta-signgd", type=float, default=0.15)
    parser.add_argument("--eta-muon", type=float, default=0.1)

    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--stop-loss", type=float, default=2e-2)

    parser.add_argument(
        "--out-csv",
        type=str,
        default="experiments/fig4_wang2509/results/decoupled_embedding_ablation_allopts.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/fig4_wang2509/figures/decoupled_embedding_ablation_allopts",
    )

    args = parser.parse_args()

    cfg = AblationConfig(K=args.K, L=args.L, block_size=args.block_size)

    df = run_all(
        cfg=cfg,
        eta_gd=args.eta_gd,
        eta_signgd=args.eta_signgd,
        eta_muon=args.eta_muon,
        steps=args.steps,
        stop_loss=args.stop_loss,
    )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[done] wrote {args.out_csv}")

    os.makedirs(args.out_dir, exist_ok=True)

    for optimizer in ["gd", "signgd", "muon"]:
        plot_optimizer(
            df,
            optimizer=optimizer,
            out_path=os.path.join(args.out_dir, f"{optimizer}_multi_step.png"),
        )

    print_summary(df)


if __name__ == "__main__":
    main()
