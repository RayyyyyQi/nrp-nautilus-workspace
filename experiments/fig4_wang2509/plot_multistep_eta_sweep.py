import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


STEPS = [10, 50, 100, 200, 500]
COLORS = {
    10: "#440154",
    50: "#3b528b",
    100: "#21918c",
    200: "#5ec962",
    500: "#fde725",
}
LINESTYLES = {10: "-", 50: "-", 100: "-", 200: "--", 500: "-."}
ZORDERS = {10: 2, 50: 3, 100: 4, 200: 6, 500: 7}


def plot_six(csv_path, out_dir, loss_floor=1e-16):
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    for optimizer in ("gd", "signgd", "muon"):
        for regime in ("decoupled", "coupled"):
            fig, ax = plt.subplots(figsize=(8.2, 5.4))
            panel = df[(df["optimizer"] == optimizer) & (df["regime"] == regime)]

            available_steps = set(panel["steps"].unique())
            for steps in (step for step in STEPS if step in available_steps):
                sub = panel[panel["steps"] == steps].sort_values("eta")
                ax.plot(
                    sub["eta"],
                    sub["loss"].clip(lower=loss_floor),
                    label=f"{steps} steps",
                    color=COLORS[steps],
                    linestyle=LINESTYLES[steps],
                    linewidth=2.0 if steps < 200 else 2.3,
                    zorder=ZORDERS[steps],
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"Learning rate $\eta$")
            ax.set_ylabel("Population loss after fixed steps")
            ax.set_title(f"{optimizer.upper()} × {regime}: multi-step $\eta$ sweep")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(ncol=2, fontsize=9, framealpha=0.95)
            fig.tight_layout()

            out_path = os.path.join(out_dir, f"eta_loss_{optimizer}_{regime}.png")
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"[done] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="experiments/fig4_wang2509/results/multistep_eta_sweep_K300.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/fig4_wang2509/figures/7_21/local/multistep_eta_sweep_K300",
    )
    parser.add_argument("--loss-floor", type=float, default=1e-16)
    args = parser.parse_args()
    plot_six(args.csv, args.out_dir, args.loss_floor)


if __name__ == "__main__":
    main()
