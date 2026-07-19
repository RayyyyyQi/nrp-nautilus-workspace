# Figure 4 Reproduction and Diagnostics: Muon vs GD / SignGD

This directory contains my reproduction and diagnostic study of Figure 4 from:

**Muon Outperforms Adam in Tail-End Associative Memory Learning**

The goal is not only to reproduce Figure 4(b)(c), but also to investigate the
learning-rate selection, one-step versus multi-step behavior, and two
discrepancies observed during reproduction:

1. Why pure SignGD in the support-decoupled identity setting gives a horizontal \(\Delta(W)\approx 0\) curve.
2. Why Muon sometimes shows sudden jumps in \(\Delta(W)\) during multi-step optimization.

---

## 1. Original Figure 4 setting

The one-layer associative memory model is

\[
f_W(E_k)=\mathrm{softmax}(\widetilde E^\top W E_k),
\]

with population cross-entropy loss

\[
L(W)
=
-\sum_{k=1}^K p_k \log [f_W(E_k)]_k.
\]

The imbalance metric is

\[
\Delta(W)
=
\max_k [f_W(E_k)]_k
-
\min_k [f_W(E_k)]_k.
\]

Here \([f_W(E_k)]_k\) is the correct-class probability for fact \(k\).

Default parameters:

```text
K = d = 999
alpha = 0.8
beta = 0.2
L = 200
dtype = float64
```

The probability vector is two-level:

```text
p_k = alpha / L,              for k <= L
p_k = (1 - alpha) / (K - L),  for k > L
```

The embedding settings are:

```text
support-decoupled:
E = I_K
Etilde = I_K

support-coupled:
Etilde = I_{K/3} \otimes R(3.638, 2.949, 5.218)
E      = I_{K/3} \otimes R(1.715, 0.876, 3.098)
```

Logit convention:

```text
Z = Etilde.T @ W @ E
Z[j, k] = Etilde[:, j]^T W E[:, k]
softmax over rows j for each query column k
```

Gradient convention:

```python
P = softmax(Z, dim=0)
M = (P - I) * p[None, :]
grad_W = Etilde @ M @ E.T
```

Optimizers:

```text
GD:     D = -grad
SignGD: D = -sign(grad)
Muon:   D = polar(-grad)
```

The Muon implementation uses exact SVD polar direction with an eigendecomposition fallback when SVD fails.

---

## 2. Original Figure 4(b): one-step reproduction

Run:

```bash
python3 -m experiments.fig4_wang2509.run_one_step \
  --L 200 \
  --eta-min 1e-4 \
  --eta-max 1e5 \
  --num-etas 300 \
  --out experiments/fig4_wang2509/results/one_step.csv

python3 -m experiments.fig4_wang2509.plot_fig4
```

Main local outputs:

```text
results/one_step.csv
figures/fig4b_one_step.png
```

Nautilus verification outputs:

```text
results/one_step_nautilus.csv
figures/fig4b_one_step_nautilus.png
```

The local and Nautilus one-step results match.

---

## 3. Original Figure 4(c): multi-step reproduction

Final local multi-step run:

```bash
python3 -m experiments.fig4_wang2509.run_multi_step \
  --steps 2000 \
  --stop-loss 2e-2 \
  --eta-gd 250 \
  --eta-signgd 0.15 \
  --eta-muon 0.1 \
  --out experiments/fig4_wang2509/results/multi_step_smoother36.csv
```

Main local outputs:

```text
results/multi_step_smoother36.csv
figures/fig4c_multi_step_smoother36.png
```

A clean Nautilus candidate with smaller Muon learning rate:

```text
results/multi_step_smoother36_muon0p08_nautilus.csv
figures/fig4c_multi_step_smoother36_muon0p08_nautilus.png
```

Diagnostic jump cases:

```text
results/multi_step_smoother36_nautilus_rerun1.csv
figures/fig4c_multi_step_smoother36_nautilus_rerun1.png

results/multi_step_extreme_smooth.csv
figures/fig4c_multi_step_extreme_smooth.png
```

---

## 4. SignGD support-decoupled discrepancy

### 4.1 Observation

In my reproduction, pure SignGD in the support-decoupled setting gives

\[
\Delta(W)\approx 0.
\]

This appears as a horizontal line near zero.

This is not a coding bug. Under the exact support-decoupled identity setting

\[
E=\widetilde E=I,
\]

the gradient column for fact \(k\) is scaled by \(p_k\):

\[
\nabla_{:,k}L(W)
=
p_k(P_{:,k}-e_k).
\]

Pure SignGD removes this positive scale:

\[
\operatorname{sign}(p_k(P_{:,k}-e_k))
=
\operatorname{sign}(P_{:,k}-e_k).
\]

Therefore all facts share the same SignGD dynamics, which implies

\[
[f_W(E_1)]_1
=
[f_W(E_2)]_2
=
\cdots
=
[f_W(E_K)]_K
\]

and hence

\[
\Delta(W)=0.
\]

This agrees with the theoretical intuition in Appendix D, Step 3 of the paper: under the support-decoupled identity setting, all triplets share the same dynamics and SignGD achieves balanced learning.

---

## 5. SignGD decoupled embedding ablation

To test whether the horizontal SignGD curve is caused by support decoupling alone, I tested three support-decoupled embedding types.

```text
A. identity_decoupled

   E = I
   Etilde = I

B. equal_block_decoupled

   E and Etilde are tall matrices with disjoint blocks.
   Every fact uses the same local coordinate pattern.

C. hetero_block_decoupled

   E and Etilde are still support-disjoint globally.
   But different facts use different local coordinate geometry.
```

For example, with \(K=3\) and block size \(3\),

\[
E\in\mathbb{R}^{9\times 3}.
\]

The columns are globally support-disjoint:

```text
fact 1 support: coordinates 1,2,3
fact 2 support: coordinates 4,5,6
fact 3 support: coordinates 7,8,9
```

Thus \(E^\top E=I\), but \(E\) is a tall matrix, not a square orthogonal matrix.

The key result:

```text
identity_decoupled:
  SignGD Delta ≈ 0

equal_block_decoupled:
  SignGD Delta ≈ 0

hetero_block_decoupled:
  SignGD Delta becomes large and GD-like
```

Interpretation:

```text
Support decoupling alone is not sufficient to guarantee SignGD balance.

Pure SignGD is coordinate-wise and is sensitive to local L1 coordinate geometry.

The horizontal SignGD-decoupled curve comes from identity / symmetric decoupled geometry, not from support decoupling alone.
```

---

## 6. GD / Muon comparison under tall decoupled embeddings

I also tested GD and Muon under the same tall support-decoupled embeddings.

Expected and observed behavior:

```text
GD:
  identity_decoupled ≈ equal_block_decoupled ≈ hetero_block_decoupled

Muon:
  identity_decoupled ≈ equal_block_decoupled ≈ hetero_block_decoupled

SignGD:
  identity_decoupled ≈ equal_block_decoupled
  hetero_block_decoupled differs strongly
```

Reason:

GD and Muon are invariant to these tall orthonormal-column representations in logit space, as long as

\[
E^\top E=I,
\qquad
\widetilde E^\top \widetilde E=I.
\]

SignGD is not invariant because

\[
\operatorname{sign}(\widetilde E M E^\top)
\]

depends on the ambient coordinate representation.

This supports the interpretation that the original SignGD discrepancy is about coordinate-wise geometry, not about the loss or probability vector alone.

### Scripts

```text
run_signgd_decoupled_ablation.py
run_decoupled_embedding_ablation_allopts.py
plot_decoupled_embedding_ablation_allopts_onefig.py
```

### Main outputs

```text
results/signgd_decoupled_ablation.csv
figures/signgd_decoupled_ablation_one_step.png
figures/signgd_decoupled_ablation_multi_step.png

results/decoupled_embedding_ablation_allopts.csv
figures/decoupled_embedding_ablation_allopts/
figures/decoupled_embedding_ablation_allopts_onefig.png
```

### Run commands

```bash
python3 -m experiments.fig4_wang2509.run_signgd_decoupled_ablation \
  --K 300 \
  --L 60 \
  --block-size 3 \
  --eta-min 1e-3 \
  --eta-max 1e2 \
  --num-etas 160 \
  --multi-eta 0.15 \
  --multi-steps 80 \
  --stop-loss 2e-2

python3 -m experiments.fig4_wang2509.run_decoupled_embedding_ablation_allopts \
  --K 300 \
  --L 60 \
  --block-size 3 \
  --eta-gd 250 \
  --eta-signgd 0.15 \
  --eta-muon 0.1 \
  --steps 120 \
  --stop-loss 2e-2

python3 -m experiments.fig4_wang2509.plot_decoupled_embedding_ablation_allopts_onefig
```

---

## 7. Muon jump diagnostics

### 7.1 Observation

In multi-step Muon runs, \(\Delta(W)\) sometimes shows a sudden jump, even though the population loss continues decreasing smoothly.

This is not a loss optimization failure.

The jump is better interpreted as a transient imbalance event caused by numerical sensitivity of the exact Muon polar/SVD direction near a near-degenerate singular region.

### 7.2 Mechanism

Muon uses the polar direction:

\[
-\nabla L(W_t)=U_t\Sigma_tV_t^\top,
\]

\[
D_t=U_tV_t^\top.
\]

This operation removes singular-value magnitudes and keeps singular directions.

When the gradient matrix has near-zero, repeated, or near-repeated singular values, the SVD basis may become numerically unstable. A tiny perturbation can then noticeably change the polar direction.

The imbalance metric

\[
\Delta(W)
=
\max_k c_k-\min_k c_k
\]

only depends on the best and worst facts, so it can amplify small differences in individual correct-class probabilities.

Therefore:

```text
polar/SVD direction instability
+ max-min Delta metric
= visible Delta jump
```

### 7.3 Diagnostic evidence

I added a diagnostic script that records:

```text
loss
Delta
correct_min / correct_max / correct_mean / correct_std
argmin_correct / argmax_correct
relative polar direction change
perturbation sensitivity
tail singular values
numerical rank under different thresholds
```

For `eta_muon=0.075`, the jump pattern is:

```text
step 99:
  direction_change ≈ 1.75e-01
  perturb_sensitivity ≈ 1.73e-01
  numerical rank drops from 998 to 996

step 100:
  Delta jumps from ≈ 1.22e-03 to ≈ 1.27e-02
  delta_ratio ≈ 10.38
```

For `eta_muon=0.08`, the same pattern appears later:

```text
step 125:
  direction_change ≈ 1.74e-01
  perturb_sensitivity ≈ 1.72e-01
  numerical rank drops from 998 to 997

step 126:
  Delta jumps from ≈ 2.92e-04 to ≈ 2.37e-03
  delta_ratio ≈ 8.11
```

This supports the mechanism:

```text
sensitive polar direction at step t
→ update W_{t+1}
→ Delta jump at step t+1
```

The filename `muon_nojump_eta0p08` is historical. In this diagnostic run, `eta_muon=0.08` also shows a jump, but later and smaller than the `eta_muon=0.075` case.

---

## 8. Decoupled vs coupled Muon jumps

The direct cause is the same in both regimes:

```text
exact Muon polar/SVD direction can become sensitive near near-degenerate singular regions.
```

For the square orthogonal embeddings used here, coupled and decoupled Muon are
equivalent in exact arithmetic after mapping them into the same logit space.
Their singular values should therefore agree. In finite precision, their
ambient matrix representations can nevertheless trigger different SVD or
fallback behavior near a rank-deficient region, and their active max/min facts
can then differ.

Therefore:

```text
decoupled and coupled can both jump,
but they do not have to jump at the same step,
and one may jump while the other does not.
```

The jump is not a theorem-level property and is not periodic.

It depends on:

```text
eta
precision
SVD backend
rank threshold
fallback behavior
embedding representation
whether the trajectory hits a sensitive singular region
```

A run may have no visible jump, one jump, or multiple jumps.

---

## 9. Local vs Nautilus differences

Local and Nautilus can have nearly identical loss trajectories but different \(\Delta(W)\) curves.

The main discrepancy is not in the population loss. It is in:

```text
Delta(W)
correct_min
correct_max
argmin_correct / argmax_correct
Muon polar direction
```

Reason:

```text
loss is an average metric;
Delta(W) is a max-min metric.
```

Near a polar/SVD sensitive step, tiny differences from different BLAS/LAPACK/SVD backends, CPU instructions, thread scheduling, or fallback behavior can choose different singular bases.

This can cause one run to show a visible \(\Delta(W)\) jump while another run stays smooth, even if the loss trajectories are almost identical.

---

## 10. Muon diagnostic scripts and outputs

Script:

```text
run_muon_jump_diagnostics.py
```

Main outputs:

```text
results/diagnostics/muon_jump_eta0p075.csv
results/diagnostics/muon_jump_eta0p075.png

results/diagnostics/muon_nojump_eta0p08.csv
results/diagnostics/muon_nojump_eta0p08.png
```

Run commands:

```bash
python3 -m experiments.fig4_wang2509.run_muon_jump_diagnostics \
  --K 999 \
  --L 200 \
  --eta-muon 0.075 \
  --steps 180 \
  --stop-loss 2e-2 \
  --tol 1e-12 \
  --noise-rel 1e-10 \
  --out-prefix experiments/fig4_wang2509/results/diagnostics/muon_jump_eta0p075

python3 -m experiments.fig4_wang2509.run_muon_jump_diagnostics \
  --K 999 \
  --L 200 \
  --eta-muon 0.08 \
  --steps 180 \
  --stop-loss 2e-2 \
  --tol 1e-12 \
  --noise-rel 1e-10 \
  --out-prefix experiments/fig4_wang2509/results/diagnostics/muon_nojump_eta0p08
```

---

## 11. Current interpretation

### SignGD

```text
The horizontal SignGD-decoupled curve is theoretically justified for
E = Etilde = I and pure SignGD.

The discrepancy with the paper plot likely comes from implementation or plotting details,
or from the paper curve not corresponding exactly to pure SignGD with exact identity decoupled embeddings.
```

### Muon

```text
Muon is overall balanced, but not perfectly identical across all facts at every finite-precision step.

The jump is a transient imbalance event caused by polar/SVD numerical sensitivity and amplified by Delta(W).

This does not overturn the paper's qualitative conclusion because Muon still keeps Delta(W)
much smaller than GD and coupled SignGD in the main reproduction.
```

---

## 12. Open questions

1. Is the paper's plotted SignGD exactly pure `sign(grad)`, or does it use Adam-style epsilon / normalization details?
2. Is the support-decoupled plot generated with exact identity embeddings, or with a more general disjoint-support construction?
3. Does the original Muon implementation use exact SVD polar direction, Newton-Schulz approximation, or a rank threshold different from this reproduction?
4. Did the authors observe small \(\Delta(W)\) jumps under different learning rates or backends?
5. Should \(\Delta(W)\) jumps be reported as max-min metric sensitivity rather than optimization failure?

---

## 13. Repository contents

This directory now contains:

```text
Original Figure 4 reproduction:
  run_one_step.py
  run_multi_step.py
  plot_fig4.py

Embedding / model utilities:
  config.py
  embeddings.py
  model.py
  optimizers.py

SignGD decoupled diagnostics:
  run_signgd_decoupled_ablation.py
  run_decoupled_embedding_ablation_allopts.py
  plot_decoupled_embedding_ablation_allopts_onefig.py

Muon jump diagnostics:
  run_muon_jump_diagnostics.py

Main outputs:
  figures/
  results/
```

---

## 14. Canonical 7/21 experiments

This section supersedes the earlier common-range Figure 4(b) sweep and the
manually tuned Figure 4(c) reproduction as the current report-facing result.
The earlier artifacts remain useful as historical diagnostics, but should not
be used as the main figures.

Detailed notes:

```text
ONE_STEP_CONCLUSIONS_7_21.md
MULTI_STEP_OBSERVATIONS_7_21.md
```

### 14.1 One-step eta selection

Each optimizer uses its own eta interval because GD, SignGD, and Muon use
different update normalizations. Coupled and decoupled use the same grid within
each optimizer so representation sensitivity is not tuned away.

| Optimizer | Eta interval | Observed one-step endpoint |
| --- | ---: | --- |
| GD | `[0.1, 176450.669959]` | first numerical-zero loss |
| Muon | `[1e-4, 44.138992]` | first numerical-zero loss |
| SignGD | `[1e-4, 24.026491]` | covers the coupled minimum |

The upper endpoint is the first numerical floor, not the center of a wider
range. Points beyond it only repeat `(loss, Delta)=(0,0)`.

Canonical eta-loss plots:

```text
figures/7_21/local/stage1_one_step_eta/one_step_eta_loss_gd.png
figures/7_21/local/stage1_one_step_eta/one_step_eta_loss_signgd.png
figures/7_21/local/stage1_one_step_eta/one_step_eta_loss_muon.png
```

### 14.2 Canonical Figure 4(b)

Local and Nautilus agree. GD and Muon coupled/decoupled curves coincide under
the orthogonal representation change. SignGD coupled cannot reduce its
one-step loss below approximately `0.26768`.

For SignGD with `E = Etilde = I`, pure `sign(grad)` removes the positive fact
frequency scale. Every fact receives the same correct-class margin, so

\[
\Delta(W_\eta)=0
\qquad\text{for every }\eta\ge 0.
\]

This flat curve is expected, not an implementation error.

Canonical standard and `1e-16` precision figures:

```text
figures/7_21/local/stage1_one_step_eta/
figures/7_21/nautilus/stage1_one_step_eta/
```

### 14.3 Multi-step eta selection

The K=300 sweep uses 300 log-spaced eta values in `[1e-8, 1e8]` and step
budgets `{10, 50, 100, 200}`.

| Optimizer | 10 steps | 50 steps | 100 steps | 200 steps |
| --- | ---: | ---: | ---: | ---: |
| GD | `5.4423e4` | `5.4423e4` | `5.4423e4` | `5.4423e4` |
| Muon | `23.1499` | `23.1499` | `23.1499` | `23.1499` |
| SignGD decoupled | `2.2275` | `0.44893` | `0.21434` | `0.11576` |
| SignGD coupled | `2.8500` | `0.57438` | `0.27424` | `0.13094` |

GD and Muon return the same plateau endpoint because these eta values already
reach the numerical floor in approximately one and two updates. SignGD's
selected eta changes approximately as `1 / T`.

For K=999 at 200 steps, Muon's endpoint is eta-sensitive and non-monotone near
the numerical floor. It reaches numerical zero for some eta values, but can
leave that region as eta increases. This is not accurately described as Muon
failing to converge for every eta.

Canonical eta figures:

```text
figures/7_21/nautilus/stage2_multistep_eta/
```

### 14.4 Canonical Figure 4(c)

The K=999 run uses:

| Optimizer | Eta |
| --- | ---: |
| GD | `194149.194574` |
| Muon | `22.695105` |
| SignGD | `0.13` |

These eta values are good under the terminal-loss criterion but poor for
visualizing multi-step dynamics:

```text
GD reaches the numerical floor in one update.
Muon reaches the numerical floor in two updates.
SignGD decoupled takes about 162 updates and stays balanced.
SignGD coupled takes about 195 updates and shows early imbalance followed by correction.
```

Therefore the current data distribution and model do not provide a meaningful
one-step versus multi-step gap for GD or Muon. Artificially dividing the eta by
200 would only slow movement along essentially the same direction; a meaningful
follow-up should change the data distribution or model.

Canonical Local/Nautilus standard and `1e-16` figures:

```text
figures/7_21/local/stage3_figure4c/
figures/7_21/nautilus/stage3_figure4c/
```

### 14.5 Initial-direction control

The frozen-direction experiment directly tests whether iterative direction
updates matter:

```text
GD: one frozen initial-direction update reaches loss <= 1e-16.
Muon: frozen and iterative runs both reach the floor in two updates.
Muon iterative cosine to the initial direction remains about 0.999999999.
SignGD coupled: the frozen direction stalls near loss 0.26798, while the
iterative direction reaches the target around step 195.
```

Thus the initialization direction is sufficient for GD and Muon in this setup;
SignGD coupled is the clear case where genuine multi-step direction adaptation
matters.

```text
figures/7_21/local/stage3_figure4c/initial_direction_diagnostic_K999/
```

### 14.6 Updated Muon jump study

The preserved Nautilus run contains a visible decoupled jump near step 102,
while its loss agrees with Local to approximately `2e-8`. A fresh instrumented
Local/Nautilus run did not reproduce the visible jump, but it did reproduce
backend-dependent SVD behavior near the same rank-deficient region:

```text
normal numerical rank: 998 (one structural zero singular direction)
fresh Local step 101: SVD failure and eigendecomposition fallback
fresh Local/Nautilus update difference at step 101: about 6.96e-8
next-step Delta difference: about 5.95e-13, too small for a visible jump
```

Coupled and decoupled are equivalent up to numerical precision in the fresh
run: their usual update difference is approximately `1e-14` and their maximum
prediction difference is approximately `1e-16`.

The current evidence supports a run-sensitive numerical event near a
rank-deficient polar/SVD region, not a stable loss-optimization failure. The
exact preserved jump cannot be causally replayed because the old pod's complete
SVD checkpoints and library build were not retained.

Main diagnostic figures:

```text
figures/7_21/local/muon_jump_study/02_polar_instability_near_jump.png
figures/7_21/local/muon_jump_cross_diagnostic/02_environment_direction_and_delta.png
figures/7_21/local/muon_jump_cross_diagnostic/03_coupled_decoupled_two_metrics.png
```

The Local/Nautilus `.pt` checkpoint files are approximately 213 MB and 225 MB.
They are intentionally retained locally and excluded from GitHub; the compact
CSV summaries, scripts, and figures are sufficient for the repository.
