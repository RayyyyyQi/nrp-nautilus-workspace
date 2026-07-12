import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_all_in_one_figure(csv_path, out_path):
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)

    optimizers = ["gd", "signgd", "muon"]
    titles = {
        "gd": "GD",
        "signgd": "SignGD",
        "muon": "Muon",
    }

    styles = {
        "identity_decoupled": {
            "label": "Identity decoupled",
            "linestyle": "-",
        },
        "equal_block_decoupled": {
            "label": "Equal-block decoupled",
            "linestyle": "--",
        },
        "hetero_block_decoupled": {
            "label": "Heterogeneous-block decoupled",
            "linestyle": "-.",
        },
    }

    for ax, optimizer in zip(axes, optimizers):
        subopt = df[df["optimizer"] == optimizer].copy()

        for kind, cfg in styles.items():
            sub = subopt[subopt["kind"] == kind].copy()
            sub = sub.sort_values("step")

            # Keep visible region similar to your Fig 4 style
            sub = sub[(sub["loss"] >= 5e-2) & (sub["loss"] <= 1.5e1)].copy()

            if len(sub) == 0:
                continue

            sub["delta_plot"] = sub["delta"].clip(lower=1e-7, upper=1.0)

            ax.plot(
                sub["loss"],
                sub["delta_plot"],
                label=cfg["label"],
                linestyle=cfg["linestyle"],
                linewidth=2.0,
                marker="o",
                markersize=3,
                markevery=max(1, len(sub) // 12),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlim(5e-2, 1.5e1)
        ax.set_ylim(8e-8, 1.5e0)

        ax.set_xticks([1e-1, 1e0, 1e1])
        ax.set_xticklabels([r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"])

        ax.set_yticks([1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
        ax.set_yticklabels(["1e+0", "1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6", "0"])

        ax.set_title(titles[optimizer])
        ax.set_xlabel("Population Loss")
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel(r"$\Delta(W)$")

    # Put one shared legend on top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))

    fig.suptitle("Decoupled embedding ablation: GD vs SignGD vs Muon", y=1.12, fontsize=13)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    print(f"[done] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="experiments/fig4_wang2509/results/decoupled_embedding_ablation_allopts.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/fig4_wang2509/figures/decoupled_embedding_ablation_allopts_onefig.png",
    )
    args = parser.parse_args()

    plot_all_in_one_figure(args.csv, args.out)


if __name__ == "__main__":
    main()
