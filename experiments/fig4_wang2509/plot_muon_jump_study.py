"""Create the three compact figures used in the Muon-jump investigation.

The script intentionally consumes preserved CSV files, so plotting the evidence
does not rerun the expensive K=999 optimization.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "figures" / "7_21" / "local" / "muon_jump_study"


def muon_frame(path, regime):
    df = pd.read_csv(path)
    return df[(df["optimizer"] == "muon") & (df["regime"] == regime)].copy()


def mark_jump(ax, step, label=None):
    ax.axvline(step, color="#7b2cbf", linestyle=":", linewidth=1.5, alpha=0.9)
    if label:
        ax.text(step + 0.8, 0.95, label, color="#7b2cbf",
                transform=ax.get_xaxis_transform(), va="top", fontsize=9)


def plot_environment_comparison():
    local_path = RESULTS / "multi_step_smoother36.csv"
    naut_path = RESULTS / "multi_step_smoother36_nautilus_rerun1.csv"
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    styles = {"Local": ("#2166ac", "-"), "Nautilus": ("#d73027", "--")}

    for col, regime in enumerate(("decoupled", "coupled")):
        for name, path in (("Local", local_path), ("Nautilus", naut_path)):
            df = muon_frame(path, regime)
            color, ls = styles[name]
            axes[0, col].plot(df.step, df.loss, color=color, ls=ls, lw=2,
                              label=name, zorder=3 if name == "Nautilus" else 2)
            axes[1, col].plot(df.step, df.delta.clip(lower=1e-16), color=color,
                              ls=ls, lw=2, label=name,
                              zorder=3 if name == "Nautilus" else 2)
        axes[0, col].set_title(f"Muon — support-{regime}")
        axes[0, col].set_yscale("log")
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel("Step")
        axes[0, col].grid(True, which="both", alpha=0.2)
        axes[1, col].grid(True, which="both", alpha=0.2)
        axes[0, col].legend(frameon=False)
        axes[1, col].legend(frameon=False)
    axes[0, 0].set_ylabel("Population loss")
    axes[1, 0].set_ylabel(r"$\Delta(W)$")
    mark_jump(axes[1, 0], 102, "Nautilus jump")
    fig.suptitle(r"Same $\eta=0.1$: loss agrees, but $\Delta(W)$ can diverge",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "01_local_vs_nautilus_loss_delta.png", dpi=250)
    plt.close(fig)


def plot_polar_instability():
    df = pd.read_csv(RESULTS / "diagnostics" / "muon_jump_eta0p075.csv")
    win = df[(df.step >= 90) & (df.step <= 108)]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(win.step, win.delta, "o-", color="#d73027", lw=2, ms=4,
                 label=r"$\Delta(W)$")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\Delta(W)$")
    axes[0].legend(frameon=False)

    axes[1].plot(win.step, win.direction_change, "o-", color="#2166ac", ms=4,
                 label="direction change")
    axes[1].plot(win.step, win.perturb_sensitivity, "s--", color="#f28e2b", ms=4,
                 label="perturbation sensitivity", zorder=3)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Relative change")
    axes[1].legend(frameon=False)

    axes[2].plot(win.step, win["rank_gt_1e-12"], "o-", color="#1b9e77", ms=4,
                 label=r"rank $(\sigma>10^{-12})$")
    axes[2].plot(win.step, win["rank_gt_1e-10"], "s--", color="#7570b3", ms=4,
                 label=r"rank $(\sigma>10^{-10})$")
    axes[2].set_ylabel("Numerical rank")
    axes[2].set_xlabel("Step")
    axes[2].legend(frameon=False)
    for ax in axes:
        mark_jump(ax, 99, "sensitive direction" if ax is axes[1] else None)
        mark_jump(ax, 100, "visible jump" if ax is axes[0] else None)
        ax.grid(True, which="both", alpha=0.2)
    fig.suptitle(r"Muon jump follows polar/SVD sensitivity ($K=999,\ \eta=0.075$)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "02_polar_instability_near_jump.png", dpi=250)
    plt.close(fig)


def plot_active_extremes():
    df = pd.read_csv(RESULTS / "diagnostics" / "muon_jump_eta0p075.csv")
    win = df[(df.step >= 94) & (df.step <= 106)]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(win.step, win.correct_max, "o-", color="#d73027", ms=4,
                 label="correct max")
    axes[0].plot(win.step, win.correct_min, "s--", color="#2166ac", ms=4,
                 label="correct min", zorder=3)
    axes[0].fill_between(win.step, win.correct_min, win.correct_max,
                         color="#bdbdbd", alpha=0.25, label=r"$\Delta(W)$ gap")
    axes[0].set_ylabel("Correct-class probability")
    axes[0].legend(frameon=False, ncol=3)

    axes[1].step(win.step, win.argmin_correct, where="mid", color="#2166ac",
                 lw=2, label="argmin fact")
    axes[1].step(win.step, win.argmax_correct, where="mid", color="#d73027",
                 lw=2, ls="--", label="argmax fact", zorder=3)
    axes[1].scatter(win.step, win.argmin_correct, color="#2166ac", s=25)
    axes[1].scatter(win.step, win.argmax_correct, color="#d73027", marker="s", s=25)
    axes[1].set_ylabel("Active fact index")
    axes[1].set_xlabel("Step")
    axes[1].legend(frameon=False)
    for ax in axes:
        mark_jump(ax, 100, "visible jump" if ax is axes[0] else None)
        ax.grid(True, alpha=0.2)
    fig.suptitle(r"The max–min metric exposes changes in extreme facts", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "03_active_extreme_facts.png", dpi=250)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot_environment_comparison()
    plot_polar_instability()
    plot_active_extremes()
    for path in sorted(OUT.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
