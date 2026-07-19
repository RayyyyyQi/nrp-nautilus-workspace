# Multi-step eta selection and Figure 4(c) observations

## Current eta-selection definition

For a fixed budget of `T` steps, define the selected eta from the sampled grid
as the smallest eta whose loss at step `T` reaches the numerical target
`loss <= 1e-16` (or the smallest eta on the numerical-zero loss plateau).

This definition concerns the loss measured **at step T**. It does not imply
that the trajectory first reaches the target at step T. The first-hitting time
would need to be measured separately.

The current K=999 Figure 4(c) uses at most 200 steps and stops a trajectory
once `loss <= 1e-16`:

- GD: eta `194149.1946`, selected from the K=999 200-step zero plateau.
- Muon: eta `22.6951`, selected from the K=999 200-step zero plateau.
- SignGD: shared eta `0.13` for coupled and decoupled, chosen as a fair
  compromise between their separate K=999 200-step thresholds. It is not the
  separate first-zero eta for both representations.

## Observation 1: dependence on the step budget at K=300

Using the smallest sampled eta on the final-loss zero plateau:

| Optimizer | 10 steps | 50 steps | 100 steps | 200 steps |
| --- | ---: | ---: | ---: | ---: |
| GD | `5.4423e4` | `5.4423e4` | `5.4423e4` | `5.4423e4` |
| Muon | `23.1499` | `23.1499` | `23.1499` | `23.1499` |
| SignGD decoupled | `2.2275` | `0.44893` | `0.21434` | `0.11576` |
| SignGD coupled | `2.8500` | `0.57438` | `0.27424` | `0.13094` |

SignGD's selected eta changes with the step budget and is approximately
inverse in `T`, so `T * eta` remains of a similar scale.

GD and Muon return the same selected eta for all recorded step budgets. The
current interpretation is not that their general dynamics are step-insensitive.
Rather, these eta values already drive GD and Muon to the numerical floor in
approximately one and two steps, respectively. Every later checkpoint then
observes the same saturated endpoint.

## Observation 2: non-monotone Muon endpoint behavior at K=999

For K=999 and a fixed 200-step run, GD and SignGD show broad low-loss plateaus
after their eta thresholds in the searched ranges. Muon's 200-step endpoint is
instead non-monotone in eta near the best region:

| Muon eta | Loss at step 200 |
| ---: | ---: |
| `19.4149` | `2.75e-15` |
| `20.9910` | `8.10e-17` |
| `22.6951` | numerical zero |
| `24.5375` | numerical zero |
| `26.5295` | `6.04e-10` |
| `28.6832` | `7.01e-11` |
| `31.0117` | `3.35e-11` |

Therefore, Muon is not accurately described as failing to converge for every
eta. It does reach numerical zero for some eta values. The observation is that
its fixed-200-step endpoint is eta-sensitive and non-monotone, and it can leave
the numerical-zero region as eta increases.

A working hypothesis is that Muon's polar normalization keeps a finite-scale
direction for very small but nonzero gradients. If the experiment always runs
all 200 updates, it may continue moving near a saturated solution, producing
oscillation or phase-sensitive endpoints. GD's update magnitude vanishes with
the gradient, which makes its low-loss plateau more stable. This mechanism is
not yet established by the current plots alone.

## Observation 3: the selected eta is poor for visualizing Figure 4(c)

The endpoint-based eta criterion gives excellent terminal loss but an
uninformative trajectory for GD and Muon:

- GD reaches numerical-zero loss in one step.
- Muon has loss about `2.58e-8` and `Delta(W)` about `1e-9` after its first
  step, then reaches numerical zero after its second step.
- SignGD decoupled needs about 162 steps with shared eta `0.13`, while keeping
  `Delta(W)` near the float64 noise scale (`~1e-14`).
- SignGD coupled needs about 195 steps. It first develops severe imbalance,
  with `Delta(W)` approaching one, and only later reduces the imbalance while
  continuing to reduce loss.

Consequently, the current Figure 4(c) cannot show how GD and Muon manage
imbalance throughout optimization: they terminate after only one or two
updates. SignGD coupled is the only trajectory that clearly displays the
paper-like pattern of early imbalance followed by later correction.

This does not mean the eta values are incorrectly selected under the stated
terminal-loss criterion. It means that the criterion is poorly matched to the
goal of visualizing multi-step dynamics.

## Questions for discussion with the advisor

1. Should Figure 4(c) use eta optimized for the lowest loss at a fixed
   200-step endpoint, even when GD and Muon finish in one or two updates?
2. Or should eta be trajectory-calibrated, for example by requiring the first
   hitting time of a common target loss to fall near steps 150--200?
3. If a trajectory-calibrated eta is preferable, what common target loss or
   hitting-time window should be used to avoid subjective manual tuning?
4. Should Muon stop when it first reaches a numerical loss threshold, or must
   all optimizers always execute exactly 200 updates? This choice matters
   because Muon's fixed-step endpoint is non-monotone near numerical
   saturation.
5. Is the goal of the reproduction to compare best terminal performance or to
   reproduce the qualitative loss--imbalance paths in the paper? These two
   goals lead to different eta-selection rules.

## Status for the later short update

This note is the detailed source for a later concise Chinese/English mixed
online update. The final short version should follow the advisor-update format
provided by the user and should distinguish observed facts from the working
hypothesis about Muon's non-monotone behavior.
