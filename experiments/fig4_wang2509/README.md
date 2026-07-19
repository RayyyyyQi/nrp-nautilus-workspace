# Figure 4 Reproduction: Muon, GD, and SignGD

Code, configurations, figures, and selected CSV outputs for reproducing Figure
4(b,c) from **Muon Outperforms Adam in Tail-End Associative Memory Learning**.

## 1. Setup

The model is

\[
f_W(E_k)=\operatorname{softmax}(\widetilde E^\top W E_k),
\]

with population loss and imbalance metric

\[
L(W)=-\sum_{k=1}^K p_k\log[f_W(E_k)]_k,
\qquad
\Delta(W)=\max_k[f_W(E_k)]_k-\min_k[f_W(E_k)]_k.
\]

Default parameters:

| Parameter | Value |
| --- | ---: |
| `K = d` | `999` |
| `L` | `200` |
| `alpha` | `0.8` |
| `beta` | `0.2` |
| dtype | `float64` |
| SVD tolerance | `1e-12` |

The fact probabilities are

```text
p_k = alpha / L                    for the first L facts
p_k = (1 - alpha) / (K - L)        for the remaining facts
```

Embedding regimes:

```text
support-decoupled:
  E = I_K
  Etilde = I_K

support-coupled:
  Etilde = I_(K/3) kron R(3.638, 2.949, 5.218)
  E      = I_(K/3) kron R(1.715, 0.876, 3.098)
```

Update directions:

```text
GD:     -grad
SignGD: -sign(grad)
Muon:   polar(-grad)
```

Core implementation:

```text
config.py
embeddings.py
model.py
optimizers.py
run_one_step.py
run_multi_step.py
plot_fig4.py
```

## 2. One-step eta selection

Purpose: select an optimizer-specific eta range that covers the informative
one-step loss decrease without extending far beyond the numerical loss floor.

| Optimizer | Eta range | Number of eta values |
| --- | ---: | ---: |
| GD | `[0.1, 176450.669959]` | `800` |
| Muon | `[1e-4, 44.138992]` | `800` |
| SignGD | `[1e-4, 24.026491]` | `800` |

Coupled and decoupled use the same eta grid within each optimizer.

Run locally:

```bash
python3 -m experiments.fig4_wang2509.run_one_step \
  --L 200 \
  --num-etas 800 \
  --eta-gd 0.1 176450.669959 \
  --eta-signgd 1e-4 24.026491 \
  --eta-muon 1e-4 44.138992 \
  --out experiments/fig4_wang2509/results/7_21/local/fig4b_one_step_algorithm_specific_eta.csv
```

Nautilus configuration:

```text
k8s/fig4b-eta-1e-1-1e7-0721.yaml
run_fig4b_eta_1e-1_1e7_nautilus.sh
```

Outputs:

```text
figures/7_21/local/stage1_one_step_eta/one_step_eta_loss_*.png
results/one_step_eta_loss_wide.csv
results/7_21/nautilus/one_step_eta_loss_wide_nautilus.csv
```

Observed minima from the sampled grids:

| Optimizer | Regime | Eta | One-step loss |
| --- | --- | ---: | ---: |
| GD | coupled / decoupled | `176450.669959` | numerical zero |
| Muon | coupled / decoupled | `44.138992` | numerical zero |
| SignGD | decoupled | `21.89` | numerical zero |
| SignGD | coupled | `23.66` | `0.26768` |

## 3. Figure 4(b)

Purpose: plot one-step loss against `Delta(W)` using the optimizer-specific eta
ranges above.

Two views are provided:

```text
standard:  paper-style visible range
precision: loss and Delta(W) displayed down to 1e-16
```

Canonical figures:

```text
figures/7_21/local/stage1_one_step_eta/
  fig4b_one_step_algorithm_specific_eta_normal.png
  fig4b_one_step_algorithm_specific_eta_precision1e-16.png

figures/7_21/nautilus/stage1_one_step_eta/
  fig4b_one_step_algorithm_specific_eta_normal_nautilus.png
  fig4b_one_step_algorithm_specific_eta_precision1e-16_nautilus.png
```

The Local and Nautilus CSVs agree to within approximately `1.75e-10` across
their numeric fields. SignGD support-decoupled has `Delta(W)` at the float64
noise floor. For `E = Etilde = I`, pure SignGD gives the same correct-class
probability for every fact, so `Delta(W) = 0` in exact arithmetic.

## 4. Multi-step eta selection

Purpose: compare final loss across eta and step budgets.

### K=300 sweep

| Parameter | Value |
| --- | ---: |
| `K` | `300` |
| `L` | `60` |
| eta range | `[1e-8, 1e8]` |
| eta values | `300` |
| step budgets | `{10, 50, 100, 200}` |

Observed first sampled eta on the numerical-zero plateau:

| Optimizer | 10 steps | 50 steps | 100 steps | 200 steps |
| --- | ---: | ---: | ---: | ---: |
| GD | `5.4423e4` | `5.4423e4` | `5.4423e4` | `5.4423e4` |
| Muon | `23.1499` | `23.1499` | `23.1499` | `23.1499` |
| SignGD decoupled | `2.2275` | `0.44893` | `0.21434` | `0.11576` |
| SignGD coupled | `2.8500` | `0.57438` | `0.27424` | `0.13094` |

### K=999 sweep

| Parameter | Value |
| --- | ---: |
| `K` | `999` |
| `L` | `200` |
| step budget | `200` |

Observed sampled eta values used for Figure 4(c):

| Optimizer | Eta |
| --- | ---: |
| GD | `194149.194574` |
| Muon | `22.695105` |
| SignGD | `0.13` |

Scripts and configurations:

```text
run_multistep_eta_sweep.py
plot_multistep_eta_sweep.py
plot_multistep_eta_k999_combined.py
run_multistep_eta_wide_k300_nautilus.sh
run_multistep_eta_local_k999_nautilus.sh
k8s/multistep-eta-wide-k300-0721.yaml
k8s/multistep-eta-local-k999-step200-0721.yaml
```

Outputs:

```text
figures/7_21/nautilus/stage2_multistep_eta/
results/7_21/nautilus/multistep_eta_sweep_K300_wide_300etas/
results/7_21/nautilus/multistep_eta_local_K999_step200/
```

## 5. Figure 4(c)

Purpose: plot the multi-step loss and imbalance trajectories for the selected
eta values.

| Parameter | Value |
| --- | ---: |
| `K` | `999` |
| `L` | `200` |
| maximum steps | `200` |
| stopping loss | `1e-16` |

Canonical figures:

```text
figures/7_21/local/stage3_figure4c/
  fig4c_K999_steps200_tuned_eta_normal.png
  fig4c_K999_steps200_tuned_eta_precision1e-16.png

figures/7_21/nautilus/stage3_figure4c/
  fig4c_K999_steps200_tuned_eta_normal_nautilus.png
  fig4c_K999_steps200_tuned_eta_precision1e-16_nautilus.png
```

The Local and Nautilus K=999 CSVs agree to within approximately `3.56e-15`.
With these eta values, GD reaches the numerical loss floor after one update,
Muon after two updates, SignGD decoupled after approximately 162 updates, and
SignGD coupled after approximately 195 updates.

Scripts and configurations:

```text
plot_fig4c_high_precision.py
run_fig4c_k999_steps200_nautilus.sh
k8s/fig4c-k999-steps200-tuned-eta-0721.yaml
```

## 6. Initial-direction control

Purpose: compare iterative Muon/GD/SignGD updates with trajectories that reuse
the initialization direction at every step.

| Parameter | Value |
| --- | ---: |
| `K` | `999` |
| `L` | `200` |
| steps | `200` |
| eta values | same as Figure 4(c) |

Outputs:

```text
run_initial_direction_diagnostic.py
results/7_21/local/initial_direction_diagnostic_K999/diagnostic.csv
figures/7_21/local/stage3_figure4c/initial_direction_diagnostic_K999/
```

Observed results:

```text
GD frozen direction: reaches loss <= 1e-16 after one update
Muon frozen direction: reaches loss <= 1e-16 after two updates
Muon iterative cosine to initial direction: approximately 0.999999999
SignGD coupled frozen direction: final loss approximately 0.26798
SignGD coupled iterative direction: reaches loss <= 1e-16 near step 195
```

## 7. Decoupled embedding ablation

Purpose: test pure SignGD under identity, equal-block, and heterogeneous-block
support-decoupled embeddings.

```text
run_signgd_decoupled_ablation.py
run_decoupled_embedding_ablation_allopts.py
plot_decoupled_embedding_ablation_allopts_onefig.py
```

Outputs:

```text
results/signgd_decoupled_ablation.csv
results/decoupled_embedding_ablation_allopts.csv
figures/7_14/signgd_decoupled_ablation_*.png
figures/7_14/decoupled_embedding_ablation_allopts/
```

SignGD has `Delta(W)` near zero for the identity and equal-block settings, but
not for the heterogeneous-block setting. GD and Muon remain invariant across
the tested orthonormal-column representations up to numerical precision.

## 8. Muon jump diagnostics

Purpose: record loss, `Delta(W)`, polar-direction changes, singular values, and
Local/Nautilus differences around preserved Muon jump cases.

Preserved eta diagnostics:

| Eta | Sensitive direction step | Delta jump step | Delta ratio |
| ---: | ---: | ---: | ---: |
| `0.075` | `99` | `100` | `10.38` |
| `0.08` | `125` | `126` | `8.11` |

In the preserved eta `0.1` comparison, the maximum Local/Nautilus loss
difference is approximately `1.98e-8`, while the Nautilus decoupled run shows a
visible `Delta(W)` jump near step 102. A fresh instrumented Local/Nautilus run
did not reproduce the visible jump. Its maximum direction difference near the
recorded window was approximately `6.96e-8`.

Scripts:

```text
run_muon_jump_diagnostics.py
run_muon_cross_diagnostics.py
plot_muon_jump_study.py
plot_muon_cross_diagnostics.py
k8s/muon-jump-cross-diagnostic-0721.yaml
```

Main figures:

```text
figures/7_21/local/muon_jump_study/02_polar_instability_near_jump.png
figures/7_21/local/muon_jump_cross_diagnostic/01_local_nautilus_singular_rank.png
figures/7_21/local/muon_jump_cross_diagnostic/02_environment_direction_and_delta.png
figures/7_21/local/muon_jump_cross_diagnostic/03_coupled_decoupled_two_metrics.png
```

Compact CSV summaries are included in `results/`. The Local and Nautilus `.pt`
checkpoint files are retained locally and are not committed because they are
approximately 213 MB and 225 MB.

## 9. Directory layout

```text
experiments/fig4_wang2509/
  README.md
  config.py, embeddings.py, model.py, optimizers.py
  run_*.py
  plot_*.py
  k8s/                         Nautilus job definitions
  figures/
    7_14/                      earlier reproduction and ablations
    7_21/
      local/
      nautilus/
  results/
    7_21/
      local/
      nautilus/
```
