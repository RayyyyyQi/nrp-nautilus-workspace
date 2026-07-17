import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


OPTIMIZERS = ("gd", "signgd", "muon")


def plot_eta_loss(csv_path: str, out_dir: str, loss_floor: float = 1e-16):
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    styles = {
        "coupled": {
            "label": "Coupled",
            "color": "#1f77b4",
            "linestyle": "-",
            "marker": "o",
            "zorder": 2,
            "annotation_offset": (8, 18),
        },
        "decoupled": {
            "label": "Decoupled",
            "color": "#d62728",
            "linestyle": "--",
            "marker": "X",
            "zorder": 4,
            "annotation_offset": (8, 58),
        },
    }

    for optimizer in OPTIMIZERS:
        fig, ax = plt.subplots(figsize=(8.2, 5.4))

        # Draw coupled first and decoupled second. When the curves coincide,
        # the blue solid line remains visible through gaps in the red dashes.
        for regime in ("coupled", "decoupled"):
            sub = df[
                (df["optimizer"] == optimizer) & (df["regime"] == regime)
            ].sort_values("eta")
            style = styles[regime]

            # Preserve the original float64 loss in the CSV. The floor is only
            # needed because an exact numerical zero cannot be shown on a log axis.
            loss_plot = sub["loss"].clip(lower=loss_floor)
            ax.plot(
                sub["eta"],
                loss_plot,
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                zorder=style["zorder"],
            )

            # idxmin returns the first point on a flat numerical-zero plateau,
            # which is the smallest eta attaining the recorded minimum loss.
            min_idx = sub["loss"].idxmin()
            eta_min = float(sub.loc[min_idx, "eta"])
            loss_min = float(sub.loc[min_idx, "loss"])
            loss_min_display = 0.0 if loss_min == 0.0 else loss_min
            loss_min_plot = max(loss_min, loss_floor)

            ax.scatter(
                [eta_min],
                [loss_min_plot],
                color=style["color"],
                marker=style["marker"],
                s=58,
                edgecolor="white",
                linewidth=0.8,
                zorder=style["zorder"] + 2,
            )
            ax.annotate(
                rf"{style['label']}: $\eta={eta_min:.4g}$"
                + "\n"
                + rf"$L={loss_min_display:.6g}$",
                xy=(eta_min, loss_min_plot),
                xytext=style["annotation_offset"],
                textcoords="offset points",
                fontsize=8,
                color=style["color"],
                ha="left",
                va="center",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": style["color"], "alpha": 0.85},
                arrowprops={"arrowstyle": "-", "color": style["color"], "lw": 0.8},
                zorder=style["zorder"] + 3,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Learning rate $\eta$")
        ax.set_ylabel("Population loss")
        ax.set_title(f"One-step $\eta$ sweep: {optimizer.upper()}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"one_step_eta_loss_{optimizer}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[done] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="experiments/fig4_wang2509/results/one_step_eta_loss_wide.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/fig4_wang2509/figures/7_21/local",
    )
    parser.add_argument("--loss-floor", type=float, default=1e-16)
    args = parser.parse_args()

    plot_eta_loss(args.csv, args.out_dir, args.loss_floor)


if __name__ == "__main__":
    main()
