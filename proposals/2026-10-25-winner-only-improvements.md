# Proposal: let a winner-only improvement pass the gate

**Date:** 2026-10-25 · **Needs:** a human decision (the gate is protected, CLAUDE.md section 9).

## Evidence
`strength.win_temperature = 1.3` (sharpen only who wins; the rest of the order unchanged, still one coherent
simulation) on the frozen 2019+ walk-forward window, 169 races:

| | champion | candidate |
|---|---|---|
| winner log-loss | 1.267 | **1.236** (-2.4%) |
| podium Brier | 0.0674 | 0.0675 |
| top-10 Brier | 0.1542 | 0.1542 |
| Spearman | 0.661 | 0.661 |
| bootstrap win rate | | 92% |
| composite | | +0.56% |

It fails only the composite >= +1% rule: the composite averages four targets, so a change that improves
one of them by 2.4% and leaves three unchanged can never reach +1%.

## Options
1. Keep the gate as is (the experiment is logged as rejected).
2. Add an alternative pass rule: primary target (winner log-loss) >= +2%, no target worse than 0.5%,
   bootstrap >= 85%, calibration not degraded. This would promote it.

Nothing is changed until a human edits `eval/gate.py`.
