import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


def make_prob_vector(K, L, alpha=0.8, dtype=torch.float64, device="cpu"):
    p = torch.empty(K, dtype=dtype, device=device)
    p[:L] = alpha / L
    p[L:] = (1.0 - alpha) / (K - L)
    return p


def loss_delta_and_P(Z, p):
    P = torch.softmax(Z, dim=0)
    correct = torch.diag(P)

    loss = -(p * torch.log(correct)).sum()
    delta = correct.max() - correct.min()

    return {
        "loss": loss.item(),
        "delta": delta.item(),
        "correct_min": correct.min().item(),
        "correct_max": correct.max().item(),
        "correct_mean": correct.mean().item(),
        "correct_std": correct.std(unbiased=False).item(),
        "argmin_correct": int(torch.argmin(correct).item()),
        "argmax_correct": int(torch.argmax(correct).item()),
        "P": P,
        "correct": correct,
    }


def grad_Z(P, p):
    K = P.shape[0]
    I = torch.eye(K, dtype=P.dtype, device=P.device)
    return (P - I) * p[None, :]


def polar_svd(G, tol=1e-12):
    """
    Robust partial polar factor of G.

    Primary:
        SVD polar direction.

    Fallback:
        eigendecomposition of G^T G.

    Returns:
        D: polar direction
        S: singular values / fallback sqrt eigenvalues, sorted descending
        info: diagnostic dictionary
    """
    info = {
        "polar_method": "svd",
        "svd_failed": False,
        "num_negative_s": 0,
    }

    try:
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)

        # Singular values should be nonnegative. If not, record it.
        info["num_negative_s"] = int((S < -1e-14).sum().item())

        # For rank diagnostics, clamp tiny numerical negatives.
        S_clean = torch.clamp(S, min=0.0)
        mask = S_clean > tol

        if mask.sum().item() == 0:
            D = torch.zeros_like(G)
        else:
            D = U[:, mask] @ Vh[mask, :]

        return D, S_clean, info

    except torch._C._LinAlgError:
        info["polar_method"] = "eigh_fallback"
        info["svd_failed"] = True

        A = G.T @ G
        A = 0.5 * (A + A.T)

        evals, V = torch.linalg.eigh(A)
        evals = torch.clamp(evals, min=0.0)

        # Sort descending to imitate SVD order.
        idx = torch.argsort(evals, descending=True)
        evals = evals[idx]
        V = V[:, idx]

        S = torch.sqrt(evals)
        mask = S > tol

        if mask.sum().item() == 0:
            D = torch.zeros_like(G)
        else:
            V_keep = V[:, mask]
            inv_sqrt = torch.rsqrt(evals[mask])
            D = (G @ V_keep) @ (inv_sqrt[:, None] * V_keep.T)

        return D, S, info


def singular_diagnostics(S, thresholds):
    out = {}

    S_sorted = S.detach()

    out["s_max"] = S_sorted[0].item()
    out["s_min"] = S_sorted[-1].item()

    # Smallest tail singular values.
    for idx in [1, 2, 3, 5, 10]:
        if len(S_sorted) >= idx:
            out[f"s_tail_{idx}"] = S_sorted[-idx].item()
        else:
            out[f"s_tail_{idx}"] = float("nan")

    for thr in thresholds:
        rank = int((S_sorted > thr).sum().item())
        out[f"rank_gt_{thr:g}"] = rank

        pos = S_sorted[S_sorted > thr]
        if len(pos) > 0:
            out[f"min_s_gt_{thr:g}"] = pos[-1].item()
            out[f"cond_gt_{thr:g}"] = (pos[0] / pos[-1]).item()
        else:
            out[f"min_s_gt_{thr:g}"] = float("nan")
            out[f"cond_gt_{thr:g}"] = float("nan")

    return out


def run_diagnostics(
    K,
    L,
    eta_muon,
    steps,
    stop_loss,
    tol,
    noise_rel,
    seed,
    device,
):
    torch.manual_seed(seed)

    dtype = torch.float64
    p = make_prob_vector(K=K, L=L, dtype=dtype, device=device)

    # For identity decoupled E=Etilde=I, Z=W.
    # For square orthogonal coupled embeddings, Muon logit-space singular dynamics
    # are equivalent in exact arithmetic.
    Z = torch.zeros(K, K, dtype=dtype, device=device)

    rows = []
    prev_D = None

    thresholds = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6]

    for step in range(steps + 1):
        stats = loss_delta_and_P(Z, p)
        P = stats.pop("P")

        M = grad_Z(P, p)
        G = -M

        D, S, polar_info = polar_svd(G, tol=tol)

        row = {
            "step": step,
            "eta_muon": eta_muon,
            **stats,
            **polar_info,
        }

        row.update(singular_diagnostics(S, thresholds))

        if prev_D is None:
            row["direction_change"] = float("nan")
        else:
            row["direction_change"] = (
                torch.linalg.norm(D - prev_D) / (torch.linalg.norm(prev_D) + 1e-30)
            ).item()

        # Perturbation sensitivity:
        # How much does polar(G) change under a tiny random perturbation?
        noise = torch.randn_like(G)
        noise = noise / (torch.linalg.norm(noise) + 1e-30)
        noise = noise * torch.linalg.norm(G) * noise_rel

        D_noisy, _, _ = polar_svd(G + noise, tol=tol)
        row["perturb_sensitivity"] = (
            torch.linalg.norm(D_noisy - D) / (torch.linalg.norm(D) + 1e-30)
        ).item()

        rows.append(row)

        if step >= 1 and stats["loss"] <= stop_loss:
            break

        prev_D = D
        Z = Z + eta_muon * D

    df = pd.DataFrame(rows)
    df["loss_change"] = df["loss"].diff()
    df["delta_change"] = df["delta"].diff()
    df["delta_ratio"] = df["delta"] / df["delta"].shift(1)

    return df


def plot_diagnostics(df, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    ax.plot(df["step"], df["loss"], marker="o", markersize=2)
    ax.set_yscale("log")
    ax.set_title("Step vs Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Population Loss")
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[0, 1]
    ax.plot(df["step"], df["delta"], marker="o", markersize=2)
    ax.set_yscale("log")
    ax.set_title("Step vs Delta")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\Delta(W)$")
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1, 0]
    ax.plot(df["step"], df["direction_change"], label="direction_change")
    ax.plot(df["step"], df["perturb_sensitivity"], label="perturb_sensitivity")
    ax.set_yscale("log")
    ax.set_title("Polar Direction Instability")
    ax.set_xlabel("Step")
    ax.set_ylabel("Relative size")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1, 1]
    ax.plot(df["step"], df["s_tail_1"], label="smallest singular")
    ax.plot(df["step"], df["s_tail_2"], label="2nd smallest")
    ax.plot(df["step"], df["s_tail_3"], label="3rd smallest")
    ax.plot(df["step"], df["min_s_gt_1e-12"], label="min s > 1e-12")
    ax.set_yscale("log")
    ax.set_title("Tail Singular Values")
    ax.set_xlabel("Step")
    ax.set_ylabel("Singular value")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=250)
    print(f"[done] wrote {out_png}")


def print_jump_report(df):
    print()
    print("=== Large Delta jumps: delta_ratio > 5 ===")
    jumps = df[df["delta_ratio"] > 5].copy()

    cols = [
        "step",
        "loss",
        "delta",
        "delta_ratio",
        "correct_min",
        "correct_max",
        "correct_mean",
        "correct_std",
        "argmin_correct",
        "argmax_correct",
        "direction_change",
        "perturb_sensitivity",
        "s_tail_1",
        "s_tail_2",
        "s_tail_3",
        "min_s_gt_1e-12",
        "rank_gt_1e-12",
        "rank_gt_1e-10",
        "rank_gt_1e-08",
    ]

    if len(jumps) == 0:
        print("No delta_ratio > 5 jumps found.")
    else:
        print(jumps[cols].to_string(index=False))

        print()
        print("=== Windows around jumps ===")
        for step in jumps["step"].tolist():
            lo = max(0, int(step) - 3)
            hi = int(step) + 4
            win = df[(df["step"] >= lo) & (df["step"] <= hi)].copy()
            print()
            print(f"--- around step {step} ---")
            print(win[cols].to_string(index=False))

    print()
    print("=== Top 10 direction_change ===")
    print(
        df.sort_values("direction_change", ascending=False)
        .head(10)[cols]
        .to_string(index=False)
    )

    print()
    print("=== Top 10 perturb_sensitivity ===")
    print(
        df.sort_values("perturb_sensitivity", ascending=False)
        .head(10)[cols]
        .to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--K", type=int, default=999)
    parser.add_argument("--L", type=int, default=200)
    parser.add_argument("--eta-muon", type=float, default=0.075)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--stop-loss", type=float, default=2e-2)

    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--noise-rel", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument(
        "--out-prefix",
        type=str,
        default="experiments/fig4_wang2509/results/diagnostics/muon_jump_eta0p075",
    )

    args = parser.parse_args()

    df = run_diagnostics(
        K=args.K,
        L=args.L,
        eta_muon=args.eta_muon,
        steps=args.steps,
        stop_loss=args.stop_loss,
        tol=args.tol,
        noise_rel=args.noise_rel,
        seed=args.seed,
        device=args.device,
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_csv = str(out_prefix) + ".csv"
    out_png = str(out_prefix) + ".png"

    df.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv}")

    plot_diagnostics(df, out_png)
    print_jump_report(df)


if __name__ == "__main__":
    main()
