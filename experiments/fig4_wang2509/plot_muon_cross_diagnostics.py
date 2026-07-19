"""Plot focused Local/Nautilus and coupled/decoupled Muon diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results" / "7_21"
OUT = ROOT / "figures" / "7_21" / "local" / "muon_jump_cross_diagnostic"


def load():
    local = pd.read_csv(BASE / "local/muon_jump_cross_diagnostic/diagnostic.csv")
    naut = pd.read_csv(BASE / "nautilus/muon_jump_cross_diagnostic/diagnostic.csv")
    return local, naut


def figure_spectrum(local, naut):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for col, (lo, hi, title) in enumerate(((0, 5, "Initial steps"), (97, 106, "Jump window"))):
        for env, df, color, ls in (("Local", local, "#2166ac", "-"),
                                    ("Fresh Nautilus", naut, "#d73027", "--")):
            x = df[(df.regime == "decoupled") & df.step.between(lo, hi)]
            # tail_1 is the expected structural zero; tail_2 is the smallest
            # ordinarily nonzero singular value.
            axes[0, col].plot(x.step, x.s_tail_2, marker="o", color=color,
                              ls=ls, lw=2, label=env)
            axes[1, col].plot(x.step, x["rank_1e-12"], marker="o", color=color,
                              ls=ls, lw=2, label=env)
        axes[0, col].set_yscale("log")
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("Smallest ordinarily nonzero singular value")
        axes[1, col].set_ylabel(r"Numerical rank ($\sigma>10^{-12}$)")
        axes[1, col].set_ylim(997.8, 999.2)
        axes[1, col].set_xlabel("Step")
        for ax in axes[:, col]:
            ax.grid(True, which="both", alpha=0.2)
            ax.legend(frameon=False)
    axes[1, 1].axvline(101, color="#7b2cbf", ls=":", lw=1.5)
    axes[1, 1].text(101.1, 998.9, "Local fallback", color="#7b2cbf", fontsize=9)
    fig.suptitle("Local vs fresh Nautilus: singular spectrum is nearly identical")
    fig.tight_layout()
    fig.savefig(OUT / "01_local_nautilus_singular_rank.png", dpi=250)
    plt.close(fig)


def figure_environment(local, naut):
    lc = torch.load(BASE / "local/muon_jump_cross_diagnostic/diagnostic.pt", map_location="cpu")
    nc = torch.load(BASE / "nautilus/muon_jump_cross_diagnostic/diagnostic.pt", map_location="cpu")
    steps = [5, 98, 99, 100, 101, 102, 103]
    direction_diff = []
    for step in steps:
        a = lc[f"decoupled_direction_step{step}"]
        b = nc[f"decoupled_direction_step{step}"]
        direction_diff.append(float(torch.linalg.norm(a - b) / torch.linalg.norm(a)))

    old_local = pd.read_csv(ROOT / "results/multi_step_smoother36.csv")
    old_naut = pd.read_csv(ROOT / "results/multi_step_smoother36_nautilus_rerun1.csv")
    old_local = old_local[(old_local.optimizer == "muon") & (old_local.regime == "decoupled")]
    old_naut = old_naut[(old_naut.optimizer == "muon") & (old_naut.regime == "decoupled")]
    fresh_l = local[(local.regime == "decoupled") & (local.step >= 97)]
    fresh_n = naut[(naut.regime == "decoupled") & (naut.step >= 97)]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=False)
    axes[0].plot(old_local.step, old_local.delta.clip(lower=1e-16), color="#2166ac",
                 lw=2, label="Preserved Local")
    axes[0].plot(old_naut.step, old_naut.delta.clip(lower=1e-16), color="#d73027",
                 ls="--", lw=2, label="Preserved Nautilus (jump)")
    axes[0].plot(fresh_n.step, fresh_n.delta, color="#f28e2b", marker="s",
                 lw=1.5, label="Fresh instrumented Nautilus")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\Delta(W)$")
    axes[0].axvline(102, color="#7b2cbf", ls=":", lw=1.5)
    axes[0].legend(frameon=False)
    axes[0].grid(True, which="both", alpha=0.2)

    axes[1].plot(steps, direction_diff, "o-", color="#7b2cbf", lw=2,
                 label="Fresh Local–Nautilus direction difference")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Relative Muon update difference")
    axes[1].grid(True, which="both", alpha=0.2)
    axes[1].legend(frameon=False)
    fig.suptitle("The preserved jump is backend/run-sensitive and did not recur fresh")
    fig.tight_layout()
    fig.savefig(OUT / "02_environment_direction_and_delta.png", dpi=250)
    plt.close(fig)


def figure_representation(local, naut):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    for env, df, color, ls in (("Local", local, "#2166ac", "-"),
                                ("Nautilus", naut, "#d73027", "--")):
        x = df[df.regime == "decoupled"]
        axes[0].plot(x.step, x.update_diff_coupled_decoupled.clip(lower=1e-18),
                     marker="o", color=color, ls=ls, lw=2, label=env)
        axes[1].plot(x.step, x.prediction_diff_coupled_decoupled.clip(lower=1e-18),
                     marker="s", color=color, ls=ls, lw=2, label=env)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel("Coupled–decoupled\nMuon update difference")
    axes[1].set_ylabel("Coupled–decoupled\nmax prediction difference")
    axes[1].set_xlabel("Recorded step")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle("Coupled and decoupled Muon are equivalent up to numerical precision")
    fig.tight_layout()
    fig.savefig(OUT / "03_coupled_decoupled_two_metrics.png", dpi=250)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    local, naut = load()
    figure_spectrum(local, naut)
    figure_environment(local, naut)
    figure_representation(local, naut)
    for path in sorted(OUT.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
