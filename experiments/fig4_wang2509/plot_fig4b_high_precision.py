import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_high_precision(csv_path: str, out_path: str, floor: float = 1e-16):
    df = pd.read_csv(csv_path)

    styles = {
        ("decoupled", "gd"): ("GD, De/Coupled", "#333333", "-", "o", 3),
        ("decoupled", "signgd"): ("SignGD, decoupled", "#ff7f0e", "--", "s", 4),
        ("coupled", "signgd"): ("SignGD, coupled", "#8c564b", "-", "s", 5),
        ("coupled", "muon"): ("Muon, coupled", "#1f77b4", "-", "^", 2),
        ("decoupled", "muon"): ("Muon, decoupled", "#d62728", "--", "^", 7),
    }

    # Coupled Muon is drawn before decoupled Muon so both remain visible.
    order = [
        ("decoupled", "gd"),
        ("decoupled", "signgd"),
        ("coupled", "signgd"),
        ("coupled", "muon"),
        ("decoupled", "muon"),
    ]

    fig, ax = plt.subplots(figsize=(8.4, 5.8))

    for regime, optimizer in order:
        sub = df[(df["regime"] == regime) & (df["optimizer"] == optimizer)].copy()
        sub = sub.sort_values("eta")

        # Figure 4(b) follows the one-step descent branch as eta increases.
        min_idx = sub["loss"].idxmin()
        eta_at_min = sub.loc[min_idx, "eta"]
        sub = sub[sub["eta"] <= eta_at_min].copy()

        sub["loss_plot"] = sub["loss"].clip(lower=floor)
        sub["delta_plot"] = sub["delta"].clip(lower=floor, upper=1.0)

        label, color, linestyle, marker, zorder = styles[(regime, optimizer)]
        ax.plot(
            sub["loss_plot"],
            sub["delta_plot"],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            marker=marker,
            markersize=3.5,
            markevery=max(1, len(sub) // 24),
            zorder=zorder,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(floor, 1.5e1)
    ax.set_ylim(floor, 1.5e0)
    ax.set_xlabel("Population Loss")
    ax.set_ylabel(r"$\Delta(W)$")
    ax.set_title(r"Figure 4(b): one-step optimization, $10^{-16}$ precision view")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.95)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[done] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="experiments/fig4_wang2509/results/one_step_eta_1e-1_1e7.csv",
    )
    parser.add_argument(
        "--out",
        default="experiments/fig4_wang2509/figures/7_21/local/fig4b_eta_1e-1_1e7_precision1e-16.png",
    )
    parser.add_argument("--floor", type=float, default=1e-16)
    args = parser.parse_args()
    plot_high_precision(args.csv, args.out, args.floor)


if __name__ == "__main__":
    main()
