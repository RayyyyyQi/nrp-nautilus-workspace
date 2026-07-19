# Figure 4(b) one-step conclusions for the 7/21 report

## Final experiment protocol

- Setting: `K=999`, `L=200`, float64.
- Each optimizer uses its own learning-rate interval because GD, SignGD, and
  Muon have different update normalizations. A common raw eta would not be a
  common update scale.
- Coupled and decoupled use exactly the same eta grid within each optimizer so
  representation sensitivity is not tuned away.
- Each interval covers the informative one-step descent from nearly the
  initial loss to the first numerical floor, or to the attainable minimum for
  SignGD coupled.
- Each optimizer uses 800 log-spaced eta values.

| Optimizer | Eta interval | Endpoint rationale |
| --- | ---: | --- |
| GD | `[0.1, 176450.669959]` | First numerical-zero loss from the wide scan |
| Muon | `[1e-4, 44.138992]` | First numerical-zero loss from the wide scan |
| SignGD | `[1e-4, 24.026491]` | Shared range through the coupled descent minimum |

The numerical-zero eta is an upper endpoint, not the center of a wider range.
Eta values beyond it add repeated `(loss, delta)=(0,0)` points and do not add
information about the optimization path.

## Observed results

- GD coupled and decoupled coincide, as expected from invariance to the
  orthogonal representation change used here.
- Muon coupled and decoupled also coincide.
- Over their common loss range, Muon has lower imbalance than GD after one
  step. This is an empirical statement for this synthetic experiment, not a
  claim that Muon is universally better on arbitrary data.
- SignGD is representation-sensitive:
  - decoupled reaches numerical-zero loss near eta `21.892`;
  - coupled cannot reduce the one-step loss below about `0.26768`, with its
    minimum near eta `23.657`.
- SignGD decoupled keeps `Delta(W)` near the float64 numerical-noise scale
  (roughly `1e-14` in the precision view). In the standard plot, values below
  `1e-7` are displayed at the fake-zero baseline.
- GD, Muon, and SignGD coupled all exhibit nonzero one-step imbalance over a
  substantial part of the loss range. Muon is less imbalanced than GD and
  SignGD coupled at comparable loss in the observed overlap.
- As GD, Muon, and SignGD decoupled approach very small loss, their imbalance
  also collapses toward numerical zero. SignGD coupled never accesses that
  small-loss region in one step.

## Interpretation of the standard plot boundary

SignGD coupled does not reach the left boundary for a substantive reason: its
minimum attainable one-step loss is about `0.26768`.

GD not touching the plotted left boundary is different. GD does reach
numerical-zero loss in the data, but the standard plotting routine discards
loss below `1e-2` and does not project an endpoint onto the visible left bound
at `5e-2`. This is a plotting/sampling boundary effect, not evidence that GD
cannot reach small loss and not a consequence of omitting small eta. Small eta
corresponds to the high-loss, right-hand side of the curve.

## Canonical artifacts

Only the new algorithm-specific-eta standard and precision figures should be
used for the 7/21 report, GitHub, and the later Notion update. Earlier common
wide-range Figure 4(b) images are superseded.
