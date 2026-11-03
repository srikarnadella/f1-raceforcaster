# Proposal: neural listwise ranker as a challenger (requested by the project owner)

**Date:** 2026-10-25 · **Status:** built, running as a live challenger; promotion only through the normal gate.

## What
`src/f1/analytics/ml.py::NeuralPL`: a small MLP (linear skip + 8 tanh units, weight decay, ensemble of 3 seeds,
L-BFGS) scoring every driver, trained on the Plackett-Luce likelihood of the finishing order (top 3 stages,
same as the champion). It drops into `models.RaceModel` when a spec sets `strength.hidden`, so it reuses the DNF
model, the simulation and every target unchanged. Gradient verified in `tests/test_analytics.py`.

## Why now
CLAUDE.md section 5 defers NNs until readiness is met. The race-count criterion is met (269 races). The plateau
criterion is not (no promotions yet). The project owner explicitly asked for neural nets on 2026-10-25, which is
the human decision this proposal records. The design rationale is the one section 5 names: a listwise model that
scores the whole field at once, where interactions (e.g. a fast car starting further back) can be non-linear.

## Safeguards (unchanged rules)
- Hyperparameters fixed in advance; not tuned on test races.
- Evaluated only with the frozen walk-forward harness (`scripts/model_lab.py`); `eval/` untouched.
- Logged live each weekend as `role: challenger` (`lab:nn+practice`), so it builds a timestamped record.
- Becomes champion only if it passes `eval/gate.py` on the full 2019+ window, like any candidate.

## Decision needed
None to keep it running as a challenger. Promotion, if the gate ever passes, follows the normal weekly cycle.
