# 7/21 meeting: one-step eta selection

This meeting artifact scans the one-step learning rate before choosing the
display interval for the Figure 4(b) reproduction.

## Fixed experiment settings

- `K = d = 999`
- `L = 200`, `alpha = 0.8`
- `float64`, CPU
- `eta` range: `[1e-8, 1e8]`
- 2,000 log-spaced learning rates
- GD, SignGD, and Muon
- decoupled and coupled embeddings

The CSV retains the original float64 losses. The figures use a plotting-only
floor of `1e-16` because exact numerical zero cannot be displayed on a log axis.
For a zero-loss plateau, the marked minimum is the smallest sampled `eta` that
attains the recorded minimum.

## Reproduce locally

```bash
bash experiments/fig4_wang2509/run_one_step_eta_loss_wide.sh
```

Local figures are written to `figures/7_21/local/`.

## Reproduce on Nautilus

```bash
bash experiments/fig4_wang2509/run_one_step_eta_loss_nautilus.sh
```

The script creates the plotting ConfigMap, submits the Job, waits for it,
and downloads the CSV and three figures. Downloaded figures are written to
`figures/7_21/nautilus/`; the CSV is written to `results/7_21/nautilus/`.
