import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def label_name(regime, optimizer):
    if optimizer == "gd":
        return f"GD ({regime})"
    if optimizer == "signgd":
        return f"SignGD ({regime})"
    if optimizer == "muon":
        return f"Muon ({regime})"
    return f"{optimizer} ({regime})"


def plot_one_step(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)

    df["loss_plot"] = df["loss"].clip(lower=1e-12)
    df["delta_plot"] = df["delta"].clip(lower=1e-12)

    plt.figure(figsize=(8, 5.5))

    styles = {
        ("decoupled", "gd"): {
            "label": "GD, decoupled",
            "color": "#1f77b4",
            "linestyle": "--",
            "marker": "o",
        },
        ("coupled", "gd"): {
            "label": "GD, coupled",
            "color": "#1f77b4",
            "linestyle": "-",
            "marker": "o",
        },
        ("decoupled", "signgd"): {
            "label": "SignGD, decoupled",
            "color": "#ff7f0e",
            "linestyle": "--",
            "marker": "s",
        },
        ("coupled", "signgd"): {
            "label": "SignGD, coupled",
            "color": "#d62728",
            "linestyle": "-",
            "marker": "s",
        },
        ("decoupled", "muon"): {
            "label": "Muon, decoupled",
            "color": "#2ca02c",
            "linestyle": "--",
            "marker": "^",
        },
        ("coupled", "muon"): {
            "label": "Muon, coupled",
            "color": "#9467bd",
            "linestyle": "-",
            "marker": "^",
        },
    }

    order = [
        ("decoupled", "gd"),
        ("coupled", "gd"),
        ("decoupled", "signgd"),
        ("coupled", "signgd"),
        ("decoupled", "muon"),
        ("coupled", "muon"),
    ]

    for regime, optimizer in order:
        sub = df[(df["regime"] == regime) & (df["optimizer"] == optimizer)]
        sub = sub.sort_values("eta")

        style = styles[(regime, optimizer)]

        plt.plot(
            sub["loss_plot"],
            sub["delta_plot"],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.6,
            marker=style["marker"],
            markersize=4,
            markevery=30,
            alpha=0.95,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Population Loss")
    plt.ylabel(r"$\Delta(W)$")
    plt.title("Figure 4(b) reproduction: one-step optimization")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8, framealpha=0.9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250)
    print(f"[done] wrote figure to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-step-csv", type=str, default="experiments/fig4_wang2509/results/one_step.csv")
    parser.add_argument("--one-step-out", type=str, default="experiments/fig4_wang2509/figures/fig4b_one_step.png")
    args = parser.parse_args()

    plot_one_step(args.one_step_csv, args.one_step_out)


if __name__ == "__main__":
    main()
