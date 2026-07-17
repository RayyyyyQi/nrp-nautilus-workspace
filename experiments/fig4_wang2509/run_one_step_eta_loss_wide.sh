#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-fig4-wang2509"

python3 -m experiments.fig4_wang2509.run_one_step \
  --L 200 \
  --eta-min 1e-8 \
  --eta-max 1e8 \
  --num-etas 2000 \
  --out experiments/fig4_wang2509/results/one_step_eta_loss_wide.csv

python3 -m experiments.fig4_wang2509.plot_one_step_eta_loss \
  --csv experiments/fig4_wang2509/results/one_step_eta_loss_wide.csv \
  --out-dir experiments/fig4_wang2509/figures/7_21/local
