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

    plt.figure(figsize=(8, 5.5))

    # Visible plotting range.
    # We want the left edge to start at 5 x 10^-2, but we do NOT show that as a tick.
    x_min_data = 1e-2
    x_left_bound = 5e-2
    x_max = 1.5e1

    # y-axis cannot show true 0 on log scale.
    # Use 1e-7 as a fake zero position and label it as "0".
    y_min = 1e-7
    y_max = 1.5e0

    styles = {
        # GD: merged line because decoupled/coupled coincide.
        ("gd_merged", "gd"): {
            "label": "GD, De/Coupled",
            "color": "#333333",
            "linestyle": "-",
            "marker": "o",
            "linewidth": 2.1,
            "alpha": 1.0,
            "zorder": 9,
        },

        # SignGD
        ("decoupled", "signgd"): {
            "label": "SignGD, decoupled",
            "color": "#ff7f0e",
            "linestyle": "--",
            "marker": "s",
            "linewidth": 2.0,
            "alpha": 1.0,
            "zorder": 4,
        },
        ("coupled", "signgd"): {
            "label": "SignGD, coupled",
            "color": "#8c564b",
            "linestyle": "-",
            "marker": "s",
            "linewidth": 2.0,
            "alpha": 1.0,
            "zorder": 5,
        },

        # Muon
        # Coupled blue is drawn first; decoupled red dashed is drawn later on top.
        ("coupled", "muon"): {
            "label": "Muon, coupled",
            "color": "#1f77b4",
            "linestyle": "-",
            "marker": "^",
            "linewidth": 1.7,
            "alpha": 1.0,
            "zorder": 2,
        },
        ("decoupled", "muon"): {
            "label": "Muon, decoupled",
            "color": "#d62728",
            "linestyle": "--",
            "marker": "^",
            "linewidth": 2.1,
            "alpha": 1.0,
            "zorder": 7,
        },
    }

    def prepare_branch(sub):
        sub = sub.sort_values("eta").copy()

        # Keep only the descent branch up to the eta that minimizes loss.
        idx_min = sub["loss"].idxmin()
        eta_min_loss = sub.loc[idx_min, "eta"]
        sub = sub[sub["eta"] <= eta_min_loss].copy()

        # Keep only visible positive-loss range for plotting.
        sub = sub[(sub["loss"] >= x_min_data) & (sub["loss"] <= x_max)].copy()

        # Log y cannot show true 0, so clip only Delta.
        sub["loss_plot"] = sub["loss"]
        sub["delta_plot"] = sub["delta"].clip(lower=y_min, upper=1e0)

        return sub

    # 1. Plot merged GD line using decoupled GD only.
    gd_sub = df[(df["regime"] == "decoupled") & (df["optimizer"] == "gd")]
    gd_sub = prepare_branch(gd_sub)

    style = styles[("gd_merged", "gd")]
    plt.plot(
        gd_sub["loss_plot"],
        gd_sub["delta_plot"],
        label=style["label"],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        marker=style["marker"],
        markersize=4,
        markevery=12,
        alpha=style["alpha"],
        zorder=style["zorder"],
    )

    # 2. Plot SignGD and Muon.
    # Important order:
    # - Muon coupled blue first
    # - Muon decoupled red dashed later, with higher zorder
    order = [
        ("decoupled", "signgd"),
        ("coupled", "signgd"),
        ("coupled", "muon"),
        ("decoupled", "muon"),
    ]

    for regime, optimizer in order:
        sub = df[(df["regime"] == regime) & (df["optimizer"] == optimizer)]
        sub = prepare_branch(sub)

        style = styles[(regime, optimizer)]

        plt.plot(
            sub["loss_plot"],
            sub["delta_plot"],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=4,
            markevery=12,
            alpha=style["alpha"],
            zorder=style["zorder"],
        )

    plt.xscale("log")
    plt.yscale("log")

    # Axis limits with a little buffer.
    plt.xlim(x_left_bound, x_max)
    plt.ylim(8e-8, y_max)

    # Only show these three x ticks, in 10^k format.
    plt.xticks(
        [1e-1, 1e0, 1e1],
        [r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"],
    )

    # Paper-style y-axis labels:
    # 1e+0, 1e-1, ..., 1e-6, 0
    plt.yticks(
        [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        ["1e+0", "1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6", "0"],
    )

    plt.xlabel("Population Loss")
    plt.ylabel(r"$\Delta(W)$")
    plt.title("Figure 4(b) reproduction: one-step optimization")

    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8, framealpha=0.95)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250)
    print(f"[done] wrote figure to {out_path}")
    
    
def plot_multi_step(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5.5))

    # Match one-step x-axis style:
    # left boundary is 5 x 10^-2, but not shown as a tick.
    x_min = 5e-2
    x_max = 1.5e1

    # Fake-zero for log-scale y-axis.
    y_min = 1e-7
    y_max = 1.5e0

    styles = {
        # GD: merged line because decoupled/coupled coincide.
        ("gd_merged", "gd"): {
            "label": "GD, De/Coupled",
            "color": "#333333",
            "linestyle": "-",
            "marker": "o",
            "linewidth": 2.1,
            "alpha": 1.0,
            "zorder": 3,
        },

        # SignGD
        ("decoupled", "signgd"): {
            "label": "SignGD, decoupled",
            "color": "#ff7f0e",
            "linestyle": "--",
            "marker": "s",
            "linewidth": 2.0,
            "alpha": 1.0,
            "zorder": 4,
        },
        ("coupled", "signgd"): {
            "label": "SignGD, coupled",
            "color": "#8c564b",
            "linestyle": "-",
            "marker": "s",
            "linewidth": 2.0,
            "alpha": 1.0,
            "zorder": 5,
        },

        # Muon
        # Coupled blue is drawn first; decoupled red dashed is drawn later on top.
        ("coupled", "muon"): {
            "label": "Muon, coupled",
            "color": "#1f77b4",
            "linestyle": "-",
            "marker": "^",
            "linewidth": 1.7,
            "alpha": 1.0,
            "zorder": 2,
        },
        ("decoupled", "muon"): {
            "label": "Muon, decoupled",
            "color": "#d62728",
            "linestyle": "--",
            "marker": "^",
            "linewidth": 2.1,
            "alpha": 1.0,
            "zorder": 7,
        },
    }

    def prepare_curve(sub):
        sub = sub.sort_values("step").copy()

        # Keep all points up to x_max first.
        sub = sub[sub["loss"] <= x_max].copy()

        # Points inside visible range.
        visible = sub[sub["loss"] >= x_min].copy()

        # If trajectory goes below x_min, keep exactly one endpoint,
        # and display it at x_min. This makes the curve touch the left boundary
        # without creating a vertical stack.
        low = sub[sub["loss"] < x_min].copy()

        if len(low) > 0:
            endpoint = low.sort_values("step").iloc[[0]].copy()
            endpoint["loss"] = x_min
            sub = pd.concat([visible, endpoint], ignore_index=True)
        else:
            sub = visible

        sub = sub.sort_values("step").copy()

        sub["loss_plot"] = sub["loss"]
        sub["delta_plot"] = sub["delta"].clip(lower=y_min, upper=1e0)

        return sub
    
    # 1. Plot merged GD line using decoupled GD only.
    # Coupled GD is numerically the same, so we do not draw it twice.
    gd_sub = df[(df["regime"] == "decoupled") & (df["optimizer"] == "gd")]
    gd_sub = prepare_curve(gd_sub)

    style = styles[("gd_merged", "gd")]
    plt.plot(
        gd_sub["loss_plot"],
        gd_sub["delta_plot"],
        label=style["label"],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=1.6,
        marker=style["marker"],
        markersize=4.5,
        markevery=3,
        alpha=style["alpha"],
        zorder=style["zorder"],
    )

    # 2. Plot SignGD and Muon.
    # Important order:
    # - Muon coupled blue first
    # - Muon decoupled red dashed later, with higher zorder
    order = [
        ("decoupled", "signgd"),
        ("coupled", "signgd"),
        ("coupled", "muon"),
        ("decoupled", "muon"),
    ]

    for regime, optimizer in order:
        sub = df[(df["regime"] == regime) & (df["optimizer"] == optimizer)]
        sub = prepare_curve(sub)

        style = styles[(regime, optimizer)]

        plt.plot(
            sub["loss_plot"],
            sub["delta_plot"],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=4,
            markevery=3,
            alpha=style["alpha"],
            zorder=style["zorder"],
        )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlim(x_min, x_max)
    plt.ylim(8e-8, y_max)

    # Only show these three x ticks.
    plt.xticks(
        [1e-1, 1e0, 1e1],
        [r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"],
    )

    # Paper-style y-axis labels.
    # The bottom "0" is actually plotted at 1e-7 because log scale cannot show true 0.
    plt.yticks(
        [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        ["1e+0", "1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6", "0"],
    )

    plt.xlabel("Population Loss")
    plt.ylabel(r"$\Delta(W)$")
    plt.title("Figure 4(c) reproduction: multi-step optimization")

    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8, framealpha=0.95)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250)
    print(f"[done] wrote figure to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-step-csv", type=str, default="experiments/fig4_wang2509/results/one_step.csv")
    parser.add_argument("--one-step-out", type=str, default="experiments/fig4_wang2509/figures/fig4b_one_step.png")
    
    parser.add_argument(
        "--multi-step-csv",
        type=str,
        default="experiments/fig4_wang2509/results/multi_step_eta1_stop1e-2.csv",
    )
    parser.add_argument(
        "--multi-step-out",
        type=str,
        default="experiments/fig4_wang2509/figures/fig4c_multi_step.png",
    )
    
    args = parser.parse_args()

    plot_one_step(args.one_step_csv, args.one_step_out)
    plot_multi_step(args.multi_step_csv, args.multi_step_out)


if __name__ == "__main__":
    main()
