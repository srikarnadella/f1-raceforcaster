# Model changelog

Every promotion, rejection and rollback. Newest last.

## 2026-10-25 — v1 initial champion
- Plackett-Luce strength (depth 3, L2 1.0, season half-life 3) + logistic DNF model + 10k-sample order simulation.
- Depth chosen on 2019–2022 only (depth 20 → 3: winner log-loss 1.601 → 1.388 on that window).
- Walk-forward 2019–2026 (169 races): winner log-loss 1.267 (grid baseline 1.445), podium Brier 0.0674 (0.0686), top-10 Brier 0.1542 (0.1633), Spearman 0.661 (0.647).
- 2026 only (17 races): worse than grid baseline on log-loss (1.271 vs 1.139). Stale pre-2026 form is the leading hypothesis.

## 2026-10-26 — simulator-blend-50: promoted as v2
- Change: `{"simulator": {"w_quali": 0.6, "lap_noise": 0.35, "c0": -1.0, "c1": 1.5, "c2": 3.0, "c3": 0.15, "start_noise": 0.7, "pit_jitter": 2.5, "pace_unc": 0.3, "team_share": 0.6, "n_sims": 1000, "weight": 0.5}}` vs champion v1
- Hypothesis: Owner objective is P1 then podium. Blend the champion 50/50 with the lap-by-lap race simulator (driver/car profiles, overtaking, strategy, safety cars; params tuned on 2023-2024 only, reports/race_sim_lab.json). podium_lab: composite +2.3% (2019+), +5.9% on 2025-2026 (out of sample for the simulator).
- Composite +2.31%, bootstrap win rate 97%, checks {'composite': True, 'no_target_regression': True, 'bootstrap': True, 'calibration': True, 'leakage': True}
- Before → after: winner_logloss 1.2672 → 1.2354, podium_brier 0.0674 → 0.0661

## 2026-10-26 — simulator-blend-30: passed gate, not chosen (one promotion per cycle)
- Change: `{"simulator": {"w_quali": 0.6, "lap_noise": 0.35, "c0": -1.0, "c1": 1.5, "c2": 3.0, "c3": 0.15, "start_noise": 0.7, "pit_jitter": 2.5, "pace_unc": 0.3, "team_share": 0.6, "n_sims": 1000, "weight": 0.3}}` vs champion v1
- Hypothesis: As simulator-blend-50 with 30% weight on the simulator (podium_lab: +2.1%, bootstrap 100%).
- Composite +2.14%, bootstrap win rate 100%, checks {'composite': True, 'no_target_regression': True, 'bootstrap': True, 'calibration': True, 'leakage': True}
- Before → after: winner_logloss 1.2672 → 1.2385, podium_brier 0.0674 → 0.0661

## 2026-10-26 — win-temperature-1.3: passed gate, not chosen (one promotion per cycle)
- Change: `{"strength": {"win_temperature": 1.3}}` vs champion v1
- Hypothesis: Favourites given 45-60% win 71% of the time (reports/winner_analysis.json); sharpen only the first-place draw.
- Composite +1.38%, bootstrap win rate 93%, checks {'composite': True, 'no_target_regression': True, 'bootstrap': True, 'calibration': True, 'leakage': True}
- Before → after: winner_logloss 1.2672 → 1.2363, podium_brier 0.0674 → 0.0675

## 2026-10-26 — adaptive-win-temperature-24: promoted as v3  _(backfilled 2026-11-02 from experiments/log.jsonl; CI run did not commit the changelog)_
- Change: `{"strength": {"adaptive_win_temperature": 24}}` vs champion v2
- Hypothesis: Winner sharpness differs by era (2023/2026 predictable, 2019-22 not). Let the model pick its own winner temperature from its last 24 training races. Offline vs v2: composite +1.34%, bootstrap 93%, 2026 winner log-loss 1.019 -> 0.944.
- Composite +1.34%, bootstrap win rate 93%, checks {'bootstrap': True, 'calibration': True, 'composite': True, 'leakage': True, 'no_target_regression': True}
- Before → after: winner_logloss 1.2321 → 1.2042, podium_brier 0.0659 → 0.0659

## 2026-10-26 — simulator-optuna-params: rejected  _(backfilled 2026-11-02 from experiments/log.jsonl; CI run did not commit the changelog)_
- Change: `{"simulator": {"c0": -1.0, "c1": 2.4435718703442797, "c2": 1.0878233973928328, "c3": 0.15, "lap_noise": 0.24563517027959714, "n_sims": 1000, "pace_unc": 0.21114146597167538, "pit_jitter": 2.5, "start_noise": 0.9033727719074526, "team_share": 0.6, "w_quali": 0.7461635386292644, "weight": 0.5}}` vs champion v2
- Hypothesis: Optuna (TPE, 40 trials, tuned on 2023-2024 only) found better simulator settings. Offline vs v2: 2026 winner log-loss 1.019 -> 0.933, 2025-26 1.057 -> 1.006, podium Brier better, top pick unchanged; composite +0.91% (bootstrap 98%).
- Composite +0.91%, bootstrap win rate 98%, checks {'bootstrap': True, 'calibration': True, 'composite': False, 'leakage': True, 'no_target_regression': True}
- Before → after: winner_logloss 1.2321 → 1.2157, podium_brier 0.0659 → 0.0657

## 2026-10-26 — era-weight-0.1: rejected  _(backfilled 2026-11-02 from experiments/log.jsonl; CI run did not commit the changelog)_
- Change: `{"strength": {"era_weight": 0.1}}` vs champion v2
- Hypothesis: Owner: separate models per era. Train mostly on the current regulation era (earlier eras weighted 0.1). Offline vs v2: 2026 winner log-loss 1.019 -> 0.954, top pick unchanged (78%); composite +0.59% alone, so best tried after the adaptive temperature.
- Composite +0.59%, bootstrap win rate 88%, checks {'bootstrap': True, 'calibration': True, 'composite': False, 'leakage': True, 'no_target_regression': True}
- Before → after: winner_logloss 1.2321 → 1.2209, podium_brier 0.0659 → 0.0658

## 2026-11-02 — simulator-tuned-2026-10-28: rejected  _(backfilled 2026-11-02 from experiments/log.jsonl; CI run did not commit the changelog)_
- Change: `{"simulator": {"c0": -1.0, "c1": 2.958372354941139, "c2": 3.3115693666092616, "c3": 0.15, "lap_noise": 0.32076082372353626, "n_sims": 1000, "pace_unc": 0.28013556586030675, "pit_jitter": 2.5, "start_noise": 1.158635381284233, "team_share": 0.6, "w_quali": 0.6482571136030112, "weight": 0.5}}` vs champion v3
- Hypothesis: Scheduled Optuna re-calibration of the race simulator on 2024-2025 (reports/race_sim_lab.json); settings {"w_quali": 0.648, "pace_unc": 0.28, "c1": 2.958, "c2": 3.312, "start_noise": 1.159, "lap_noise": 0.321}.
- Composite +0.08%, bootstrap win rate 77%, checks {'bootstrap': False, 'calibration': True, 'composite': False, 'leakage': True, 'no_target_regression': True}
- Before → after: winner_logloss 1.2090 → 1.2075, podium_brier 0.0659 → 0.0659

## 2026-11-02 — race-pace-features: rejected  _(backfilled 2026-11-02 from experiments/log.jsonl; CI run did not commit the changelog)_
- Change: `{"form": {"race_pace": true}, "strength": {"features": ["log_start", "quali_gap", "tm_quali_gap", "sprint_delta", "drv_form_pos", "con_form_pos", "drv_form_quali", "drv_gain", "drv_form_pos_early", "con_form_pos_early", "no_history", "drv_race_pace", "exp_race_gap", "con_race_minus_quali"]}}` vs champion v3
- Hypothesis: Lap-derived race pace (clean-air, fuel/tyre-corrected) and each team's race-vs-quali pace delta add signal beyond one-lap pace and start position (user suggestion; R17 showed team deltas up to 0.9%).
- Composite -0.17%, bootstrap win rate 39%, checks {'bootstrap': False, 'calibration': True, 'composite': False, 'leakage': True, 'no_target_regression': True}
- Before → after: winner_logloss 1.2090 → 1.2100, podium_brier 0.0659 → 0.0661

## 2026-11-02 — season-half-life-1.5: rejected  _(backfilled 2026-11-02 from experiments/log.jsonl; CI run did not commit the changelog)_
- Change: `{"strength": {"season_half_life": 1.5}}` vs champion v3
- Hypothesis: 2026 is a new era; down-weighting older seasons harder in the strength fit should cut stale-form error.
- Composite +0.21%, bootstrap win rate 79%, checks {'bootstrap': False, 'calibration': True, 'composite': False, 'leakage': True, 'no_target_regression': True}
- Before → after: winner_logloss 1.2090 → 1.2047, podium_brier 0.0659 → 0.0659

## 2026-11-09 — simulator-tuned-2026-11-04: rejected
- Change: `{"simulator": {"w_quali": 0.6482571136030112, "lap_noise": 0.32076082372353626, "c0": -1.0, "c1": 2.958372354941139, "c2": 3.3115693666092616, "c3": 0.15, "start_noise": 1.158635381284233, "pit_jitter": 2.5, "pace_unc": 0.28013556586030675, "team_share": 0.6, "n_sims": 1000, "weight": 0.5}}` vs champion v3
- Hypothesis: Scheduled Optuna re-calibration of the race simulator on 2024-2025 (reports/race_sim_lab.json); settings {"w_quali": 0.648, "pace_unc": 0.28, "c1": 2.958, "c2": 3.312, "start_noise": 1.159, "lap_noise": 0.321}.
- Composite +0.09%, bootstrap win rate 79%, checks {'composite': False, 'no_target_regression': True, 'bootstrap': False, 'calibration': True, 'leakage': True}
- Before → after: winner_logloss 1.2081 → 1.2065, podium_brier 0.0659 → 0.0659

## 2026-11-09 — form-half-life-4: rejected
- Change: `{"form": {"form_half_life": 4.0}}` vs champion v3
- Hypothesis: Faster-adapting driver/constructor form tracks in-season pecking-order changes better.
- Composite -0.14%, bootstrap win rate 32%, checks {'composite': False, 'no_target_regression': True, 'bootstrap': False, 'calibration': True, 'leakage': True}
- Before → after: winner_logloss 1.2081 → 1.2093, podium_brier 0.0659 → 0.0661

## 2026-11-09 — dnf-C-0.3: rejected
- Change: `{"dnf": {"C": 0.3}}` vs champion v3
- Hypothesis: Stronger regularisation of the DNF model reduces overfitting to noisy reliability history.
- Composite -0.01%, bootstrap win rate 33%, checks {'composite': False, 'no_target_regression': True, 'bootstrap': False, 'calibration': True, 'leakage': True}
- Before → after: winner_logloss 1.2081 → 1.2083, podium_brier 0.0659 → 0.0659
