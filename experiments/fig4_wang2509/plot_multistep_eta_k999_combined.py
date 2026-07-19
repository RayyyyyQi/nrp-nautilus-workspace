import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_combined(csv_path, out_path, loss_floor=1e-16):
    df = pd.read_csv(csv_path)
    df = df[df["steps"] == 200].copy()

    optimizer_styles = {
        "gd": {"label": "GD", "marker": "o"},
        "signgd": {"label": "SignGD", "marker": "s"},
        "muon": {"label": "Muon", "marker": "^"},
    }
    regime_styles = {
        "coupled": {"label": "Coupled", "linestyle": "-", "fill": True, "zorder": 2, "linewidth": 4.0},
        "decoupled": {"label": "Decoupled", "linestyle": "--", "fill": False, "zorder": 6, "linewidth": 2.2},
    }
    colors = {
        ("gd", "coupled"): "#2166ac",
        ("gd", "decoupled"): "#d73027",
        ("signgd", "coupled"): "#762a83",
        ("signgd", "decoupled"): "#f28e2b",
        ("muon", "coupled"): "#1b9e77",
        ("muon", "decoupled"): "#e7298a",
    }

    fig, ax = plt.subplots(figsize=(9.2, 6.0))

    # Solid coupled curves are drawn first; dashed decoupled curves are drawn
    # above them so coincident curves remain identifiable through dash gaps.
    for regime in ("coupled", "decoupled"):
        for optimizer in ("gd", "signgd", "muon"):
            sub = df[
                (df["optimizer"] == optimizer) & (df["regime"] == regime)
            ].sort_values("eta")
            opt = optimizer_styles[optimizer]
            reg = regime_styles[regime]
            color = colors[(optimizer, regime)]
            marker_face = color if reg["fill"] else "white"

            ax.plot(
                sub["eta"],
                sub["loss"].clip(lower=loss_floor),
                label=f"{opt['label']}, {reg['label']}",
                color=color,
                linestyle=reg["linestyle"],
                linewidth=reg["linewidth"],
                marker=opt["marker"],
                markersize=5.2,
                markerfacecolor=marker_face,
                markeredgecolor=color,
                markeredgewidth=1.1,
                markevery=4,
                zorder=reg["zorder"],
            )

            # Emphasize the smallest sampled eta attaining the minimum loss.
            min_idx = sub["loss"].idxmin()
            ax.scatter(
                [sub.loc[min_idx, "eta"]],
                [max(float(sub.loc[min_idx, "loss"]), loss_floor)],
                color=color if reg["fill"] else "white",
                edgecolor=color,
                marker=opt["marker"],
                s=78,
                linewidth=1.4,
                zorder=reg["zorder"] + 2,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Learning rate $\eta$")
    ax.set_ylabel("Population loss after 200 steps")
    ax.set_title(r"K=999 multi-step $\eta$ sweep (200 steps)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=9, framealpha=0.95)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[done] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--loss-floor", type=float, default=1e-16)
    args = parser.parse_args()
    plot_combined(args.csv, args.out, args.loss_floor)


if __name__ == "__main__":
    main()
