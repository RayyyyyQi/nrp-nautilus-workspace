import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_high_precision(csv_path, out_path, floor=1e-16):
    df = pd.read_csv(csv_path)
    styles = {
        ("gd_merged", "gd"): ("GD, De/Coupled", "#333333", "-", "o", 9),
        ("decoupled", "signgd"): ("SignGD, decoupled", "#ff7f0e", "--", "s", 4),
        ("coupled", "signgd"): ("SignGD, coupled", "#8c564b", "-", "s", 5),
        ("coupled", "muon"): ("Muon, coupled", "#1f77b4", "-", "^", 2),
        ("decoupled", "muon"): ("Muon, decoupled", "#d62728", "--", "^", 7),
    }

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    order = [
        ("decoupled", "gd", ("gd_merged", "gd")),
        ("decoupled", "signgd", ("decoupled", "signgd")),
        ("coupled", "signgd", ("coupled", "signgd")),
        ("coupled", "muon", ("coupled", "muon")),
        ("decoupled", "muon", ("decoupled", "muon")),
    ]

    for regime, optimizer, style_key in order:
        sub = df[(df["regime"] == regime) & (df["optimizer"] == optimizer)].copy()
        sub = sub.sort_values("step")
        label, color, linestyle, marker, zorder = styles[style_key]
        ax.plot(
            sub["loss"].clip(lower=floor),
            sub["delta"].clip(lower=floor),
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=8,
            markersize=4,
            linewidth=2.1 if linestyle == "--" else 1.8,
            zorder=zorder,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(floor, 2e1)
    ax.set_ylim(floor, 1.5)
    ax.set_xlabel("Population Loss")
    ax.set_ylabel(r"$\Delta(W)$")
    ax.set_title(r"Figure 4(c): 200-step optimization, $10^{-16}$ precision view")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.95)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[done] wrote figure to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--floor", type=float, default=1e-16)
    args = parser.parse_args()
    plot_high_precision(args.csv, args.out, args.floor)


if __name__ == "__main__":
    main()
