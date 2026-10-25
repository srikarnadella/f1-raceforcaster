# Model changelog

Every promotion, rejection and rollback. Newest last.

## 2026-10-25 — v1 initial champion
- Plackett-Luce strength (depth 3, L2 1.0, season half-life 3) + logistic DNF model + 10k-sample order simulation.
- Depth chosen on 2019–2022 only (depth 20 → 3: winner log-loss 1.601 → 1.388 on that window).
- Walk-forward 2019–2026 (169 races): winner log-loss 1.267 (grid baseline 1.445), podium Brier 0.0674 (0.0686), top-10 Brier 0.1542 (0.1633), Spearman 0.661 (0.647).
- 2026 only (17 races): worse than grid baseline on log-loss (1.271 vs 1.139). Stale pre-2026 form is the leading hypothesis.
