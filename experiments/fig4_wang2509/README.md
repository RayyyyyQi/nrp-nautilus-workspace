# Figure 4 Reproduction: Muon vs GD / SignGD

This directory reproduces the Figure 4 toy associative memory experiments from the paper.

The experiment studies a one-layer linear associative memory model

\[
f_W(E_k)=\mathrm{softmax}(\widetilde E^\top W E_k),
\]

with population loss

\[
L(W)=-\sum_{k=1}^K p_k\log [f_W(E_k)]_k,
\]

and imbalance metric

\[
\Delta(W)=\max_k [f_W(E_k)]_k-\min_k [f_W(E_k)]_k.
\]

The goal is to compare GD, SignGD, and Muon under support-decoupled and support-coupled embeddings.

---

## 1. Experiment setting

Default parameters:

```text
K = d = 999
alpha = 0.8
beta = 0.2
L = 200
dtype = float64
```

The probability distribution is two-level:

```text
p_k = alpha / L,              for k <= L
p_k = (1 - alpha) / (K - L),  for k > L
```

Embeddings:

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
softmax over rows j
```

Optimizers:

```text
GD:     -grad
SignGD: -sign(grad)
Muon:   polar(-grad)
```

The Muon implementation uses SVD by default. I also added an eigendecomposition fallback for numerical robustness when `torch.linalg.svd` fails to converge.

---

## 2. Figure 4(b): one-step reproduction

### Local run

```bash
python3 -m experiments.fig4_wang2509.run_one_step \
  --L 200 \
  --eta-min 1e-4 \
  --eta-max 1e5 \
  --num-etas 300 \
  --out experiments/fig4_wang2509/results/one_step.csv

python3 -m experiments.fig4_wang2509.plot_fig4
```

Output:

```text
results/one_step.csv
figures/fig4b_one_step.png
```

### Nautilus verification

Output:

```text
results/one_step_nautilus.csv
figures/fig4b_one_step_nautilus.png
```

The local and Nautilus one-step results are consistent.

---

## 3. Figure 4(b): discrepancies from the paper plot

There are two main differences compared with the paper plot.

### 3.1 SignGD, support-decoupled

In the support-decoupled setting,

\[
E=\widetilde E=I.
\]

With pure SignGD,

```text
D = -sign(grad),
```

the sign operation removes the frequency scale \(p_k\). Therefore, all facts receive almost the same sign-gradient margin, so

\[
\Delta(W)\approx 0.
\]

As a result, the SignGD support-decoupled curve appears as a horizontal line near zero, instead of forming a visible curve similar to Muon.

### 3.2 SignGD, support-coupled

I swept learning rates and also tested

```text
L = 198, 199, 200, 201
```

For pure SignGD, the best one-step coupled loss stayed around

```text
loss ≈ 0.27
```

so the curve could not extend toward loss \(0\) as in the original plot.

This suggests that the paper implementation may use a slightly different SignGD convention, normalization, or optimizer detail.

---

## 4. Figure 4(c): multi-step reproduction

Multi-step updates use

\[
W_{t+1}=W_t+\eta D_t,
\]

where \(D_t\) is recomputed at every step.

The plot uses population loss on the x-axis and \(\Delta(W)\) on the y-axis.

---

## 5. Figure 4(c): final local result

Final local version:

```text
results/multi_step_smoother36.csv
figures/fig4c_multi_step_smoother36.png
```

Command:

```bash
python3 -m experiments.fig4_wang2509.run_multi_step \
  --steps 2000 \
  --stop-loss 2e-2 \
  --eta-gd 250 \
  --eta-signgd 0.15 \
  --eta-muon 0.1 \
  --out experiments/fig4_wang2509/results/multi_step_smoother36.csv
```

Summary:

```text
GD:
  final_step = 323
  final_loss ≈ 1.993776e-02
  final_delta ≈ 8.010181e-02

SignGD, decoupled:
  final_step = 37
  final_loss ≈ 1.496950e-02
  final_delta ≈ 4.24e-14

SignGD, coupled:
  final_step = 37
  final_loss ≈ 1.634240e-02
  final_delta ≈ 9.978771e-02

Muon:
  final_step = 109
  final_loss ≈ 1.818892e-02
  final_delta ≈ 1.36e-04
```

This is the main local Figure 4(c) reproduction result.

---

## 6. Figure 4(c): Nautilus no-jump candidate

The clean Nautilus no-jump candidate uses a slightly smaller Muon learning rate:

```text
results/multi_step_smoother36_muon0p08_nautilus.csv
figures/fig4c_multi_step_smoother36_muon0p08_nautilus.png
```

Parameters:

```text
steps = 4000
stop-loss = 2e-2
eta-gd = 250
eta-signgd = 0.15
eta-muon = 0.08
```

Summary:

```text
Muon decoupled:
  final_step = 136
  final_loss ≈ 1.8553e-02
  final_delta ≈ 1.39e-04

Muon coupled:
  final_step = 136
  final_loss ≈ 1.8553e-02
  final_delta ≈ 1.39e-04
```

Jump diagnostic:

```text
decoupled Muon jumps: Empty DataFrame
coupled Muon jumps:   Empty DataFrame
```

This is the clean Nautilus verification candidate.

---

## 7. Figure 4(c): diagnostic jump cases

I kept two diagnostic jump cases because Muon sometimes shows sudden jumps in \(\Delta(W)\), even when the loss trajectory remains smooth.

### 7.1 Nautilus jump diagnostic

Files:

```text
results/multi_step_smoother36_nautilus_rerun1.csv
figures/fig4c_multi_step_smoother36_nautilus_rerun1.png
```

Parameters:

```text
steps = 2000
stop-loss = 2e-2
eta-gd = 250
eta-signgd = 0.15
eta-muon = 0.1
```

Observation:

```text
Nautilus Muon decoupled shows a jump around:
step ≈ 102
loss ≈ 0.0363
delta_ratio ≈ 9.54
```

A local run with the same hyperparameters was smooth, while Nautilus showed a jump.

Local-vs-Nautilus comparison for the same parameters:

```text
loss max_abs_diff        ≈ 1.98e-08
delta max_abs_diff       ≈ 2.32e-03
correct_min max_abs_diff ≈ 1.49e-03
correct_max max_abs_diff ≈ 8.25e-04
```

This means the population loss trajectories are essentially identical, but \(\Delta(W)\) differs because the max/min correct probabilities separate on Nautilus.

### 7.2 Local jump diagnostic

Files:

```text
results/multi_step_extreme_smooth.csv
figures/fig4c_multi_step_extreme_smooth.png
```

Parameters:

```text
steps = 6000
stop-loss = 2e-2
eta-gd = 100
eta-signgd = 0.05
eta-muon = 0.075
```

Observation:

```text
Local Muon decoupled shows a jump around:
step ≈ 100
loss ≈ 0.4386
delta_ratio ≈ 10.38
```

This suggests that the jump is not only a Nautilus issue; it can also occur locally for certain Muon learning rates.

---

## 8. Interpretation of Muon jumps

The Muon update uses the polar direction

\[
D_t=\operatorname{polar}(-\nabla L(W_t)).
\]

If

\[
-\nabla L(W_t)=U\Sigma V^\top,
\]

then Muon uses

\[
D_t=UV^\top.
\]

This operation ignores the singular values and keeps only singular vector directions.

When the gradient has repeated, very close, or near-zero singular values, the polar direction can become numerically sensitive. Different linear algebra backends can choose slightly different singular vector bases inside nearly degenerate subspaces.

The metric

\[
\Delta(W)=\max_k [f_W(E_k)]_k-\min_k [f_W(E_k)]_k
\]

is also sensitive because it depends on active max/min facts. Therefore, even if the population loss trajectory is stable, a small numerical difference in the Muon direction can produce a visible jump in \(\Delta(W)\).

Current interpretation:

```text
The qualitative Figure 4(c) trend is reproducible:
Muon keeps Delta(W) small,
GD and coupled SignGD are more imbalanced,
and decoupled SignGD remains near Delta(W)=0.

However, Muon curves can show numerical sensitivity in this toy setting,
especially under the max-min imbalance metric Delta(W).
```

---

## 9. Files kept

### Results

```text
results/one_step.csv
results/one_step_nautilus.csv

results/multi_step_smoother36.csv
results/multi_step_smoother36_muon0p08_nautilus.csv
results/multi_step_smoother36_nautilus_rerun1.csv
results/multi_step_extreme_smooth.csv
```

### Figures

```text
figures/fig4b_one_step.png
figures/fig4b_one_step_nautilus.png

figures/fig4c_multi_step_smoother36.png
figures/fig4c_multi_step_smoother36_muon0p08_nautilus.png
figures/fig4c_multi_step_smoother36_nautilus_rerun1.png
figures/fig4c_multi_step_extreme_smooth.png
```

---

## 10. Status

Current status:

```text
Figure 4(b): local + Nautilus complete.
Figure 4(c): local final complete.
Figure 4(c): Nautilus no-jump candidate complete.
Figure 4(c): diagnostic Muon jump cases preserved.
```

Main open question:

```text
Is the original paper using pure SignGD, or a normalized / Adam-like variant?
```

A second open question:

```text
How should Muon polar-direction numerical sensitivity be handled or reported
when Delta(W) is a max-min metric?
```
