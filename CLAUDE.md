# F1 Forecaster: Project Brief & Operating Instructions

Read this file fully at the start of every session. It defines what this project is, how it runs each race weekend, what you are allowed to change on your own, and what you must never touch.

## 1. Mission

Forecast Formula 1 outcomes from practice and qualifying data, score those forecasts honestly against reality, and keep improving the models across the season, autonomously but safely.

**Prediction targets (all four are in scope):**
1. **Race winner**: win probability per driver
2. **Podium and top 10**: P(finish top 3), P(finish top 10) per driver
3. **Full finishing order**: expected finishing position per driver plus a predicted order
4. **Championship standings**: title probabilities for drivers and constructors, updated every race

**Out of scope for now:** qualifying prediction, live in-race prediction, betting advice. Never present output as betting or financial advice.

## 2. Current state (update this section as things change)

_Updated 2026-10-25._ The v0 scaffold files (`f1data.py`, `train.py`, ...) never existed in this repo; the project was built directly in `src/f1/`.

- **Data source (verified on real data):** FastF1 3.9 lap/timing data now requires an F1TV subscription (paid, so not used; see `proposals/2026-10-25-f1tv-practice-data.md`). The pipeline uses the free **Jolpica** (Ergast) API: qualifying Q1–Q3, sprint, race results, race lap times, pit stops. Store: `data/results/<season>.csv` 2014–2026 (273 races), `data/calendar.csv`, `data/laps/` (race laps, 2022+ backfill).
- **Practice features:** `src/f1/practice.py` written but **unverified** (no credentials). Weekend pace is qualifying-based.
- **Champion v1:** Plackett-Luce strength (depth 3) + logistic DNF model → 10k-sample order simulation; winner, podium, top-10, order and championship all come from that one simulation (coherent; sums checked in tests).
- **Real-data baseline (walk-forward 2019–2026, 169 races):** v1 winner log-loss 1.267 (grid baseline 1.445, pole 2.001, uniform 3.005); podium Brier 0.0674 (grid 0.0686); top-10 Brier 0.1542 (grid 0.1633); Spearman 0.661 (grid 0.647); top-1 hit 56% (pole sitter 56%). **2026 only (17 races): v1 is worse than grid on winner log-loss (1.271 vs 1.139)**, likely because pre-2026 form is stale. Details in `reports/backtest*.json`.
- **Milestones:** M0–M5 built and run on real data; M6/M7 (workflow, registry, gate, shadow, rollback) built, with the gate exercised in dry run. First live forecast: 2026 R18, logged 19:39 UTC before the 20:00 start.
- **Lap analysis:** `src/f1/laps.py` tags race laps (lap 1, in/out, neutralised, slow, traffic, clean), fits fuel and tyre slopes per race, and clusters stints into inferred compound classes (the free data has no compound).

## 3. The two weekly triggers

The project runs on **GitHub Actions (scheduled)**. A cron job fires every ~2 hours Friday to Monday (UTC). A gate script (`scripts/should_run.py`) checks the FastF1 event schedule and `state/last_run.json` and decides whether either job is due. Jobs must be **idempotent**, so re-running a completed job changes nothing.

### Trigger A: Pre-race forecast (after qualifying)
Runs once all pre-race sessions are finished and data is available.
1. Ingest practice and qualifying data for the weekend.
2. Build features using only information available at this moment.
3. Run the **current champion model** and write forecasts for all four targets.
4. **Append to the immutable prediction log** (`predictions/log.jsonl`) with timestamp, model version, and feature hash, **before the race happens**. Never edit or delete entries.
5. Update the dashboard data.

Sprint weekends have a different session order, so handle them. Use whatever practice exists, and record the format as a feature or flag.

### Trigger B: Post-race review (after the race results are available)
1. Ingest the race results and race-pace data.
2. **Score** the logged prediction for this race on every target (see section 6).
3. **Write the weekend pace summary** (section 7).
4. Update championship state and recompute title probabilities.
5. Run an **improvement cycle** (section 8) within the compute/time budget.
6. Commit all outputs; the dashboard picks them up.

**Data lag:** FastF1 data sometimes appears hours after a session. If data is missing or incomplete, exit cleanly and let the next cron tick retry. If still missing by Tuesday, open a GitHub issue; don't loop forever.

## 4. Design principles

- **No leakage, ever.** Every feature must be computable from data available at prediction time. Form features use earlier races only. A leakage bug is the worst failure here, because it makes every number look better than reality.
- **The prediction log is the source of truth for performance.** Backtests guide development, but only live, timestamped predictions count as real accuracy.
- **Small data, so favour simple models and honest uncertainty.** About 24 races a year means winner accuracy is very noisy. Report calibration and confidence intervals, not just accuracy. "Pole sitter wins" is a strong baseline; always compare against it.
- **Regulation eras matter.** 2026 is a new regulation set, so pre-2026 form features go stale. Prefer field-relative pace features, include an era feature, and test down-weighting old seasons.
- **Coherent outputs.** Winner, podium, top-10, order, and championship numbers must agree with each other. The intended design: one set of pace scores per driver, a finishing-order simulation (Plackett-Luce-style sampling plus a DNF model), and all targets derived from the same simulation. Start with simple separate models if needed, but check that marginals are consistent and treat coherence as an upgrade goal.
- **Reproducibility.** Fixed seeds, pinned dependencies, every model artifact tied to a git commit and data snapshot.

## 5. Modelling scope

**Classical first.** Logistic regression, gradient boosting (HistGradientBoosting/LightGBM), ordinal or Plackett-Luce ranking models, Elo or Bradley-Terry style driver and car ratings, and calibration (isotonic or Platt) are all in scope.

**Neural networks are deferred.** Do not build them yet. Instead, at the end of each improvement cycle, evaluate and log a one-line **NN readiness check**:
- Labeled races available (target: **150+**, or lap-level features adopted that multiply the data)
- Whether classical models have plateaued (last 3 promotions each improved < 1%)
- Whether a specific NN design (e.g., a listwise ranking model that scores the whole field at once) has a clear rationale

When readiness is met, **propose** the NN as a *challenger only* in `proposals/` (see section 9). It must beat the classical champion through the normal promotion gate and gets no special treatment.

**Starter backlog** (roughly in priority order; Claude may reorder based on evidence):
1. Verify real data pipeline; record baselines
2. DNF/reliability model (needed for order and championship)
3. Podium and top-10 models, finishing-order model
4. Championship Monte Carlo simulation (remaining calendar, points system including sprints, drivers and constructors)
5. Long-run pace corrections for tyre compound and fuel load
6. Teammate-relative pace, car-vs-driver ratings
7. Circuit-type features, weather/wet-session flag, grid penalties
8. Sprint-weekend features
9. Calibration and ensembling

## 6. Metrics

| Target | Metrics |
|---|---|
| Winner | log-loss (primary), Brier, top-1 hit, winner's rank |
| Podium / top 10 | Brier score per driver, calibration curve |
| Finishing order | Spearman and Kendall rank correlation, mean absolute position error |
| Championship | ranked probability score of final standings, evaluated at checkpoints (e.g., after races 5, 10, 15, 20) |

Always report against baselines: pole sitter wins, grid-order finish, and a uniform guess. Evaluate with **walk-forward** splits only (train on earlier races, test on the next). Never random k-fold.

## 7. Weekend pace summary (Trigger B output)

Save to `reports/<year>_<round>_pace.md` and surface it in the dashboard. Include:
- **Practice vs. qualifying vs. race pace** by team and driver: best-lap gaps, long-run pace, quali gaps, race clean-air median pace
- **Who over/under-delivered** relative to what practice and qualifying suggested
- **Prediction review**: what we forecast vs. what happened, the biggest misses, and a *plausible, evidence-based* explanation for each (safety car, strategy, weather, DNF, penalty). Label unsure explanations as hypotheses, never as facts.
- **Takeaways for the model**: concrete feature or data issues the weekend exposed (feeds the improvement cycle)

## 8. The improvement cycle (autonomous)

Run at the end of Trigger B, within a budget of **≤ 3 experiments and ≤ 30 minutes of compute per cycle**. Do a deeper review at season end.

1. **Diagnose.** Review the latest scores, calibration, and biggest misses. Pick the highest-value hypothesis from the backlog or from the pace summary.
2. **Experiment** on a branch (`exp/<date>-<slug>`). Change one thing at a time where possible. Evaluate with the frozen walk-forward harness.
3. **Gate** (all must hold to promote):
   - Composite skill vs. the champion (mean relative improvement across winner log-loss, podium Brier, top-10 Brier, order Spearman) is **≥ +1%**
   - **No single target worse by more than 2%**
   - Paired bootstrap over races shows the candidate beats the champion in **≥ 80%** of resamples
   - Calibration not degraded
   - No leakage check failures (the leakage tests in `tests/` pass)
4. **Promote** if the gate passes: merge, register the model, tag the version, log it. Otherwise record the failed experiment anyway, since negative results prevent repeat work.
5. **Shadow and roll back.** The previous champion keeps running in **shadow mode** for 6 live races. If the new champion's live rolling composite is worse than the old one's by more than 2%, **auto-rollback** and log why.
6. **Season-end review.** Full retrain, feature ablations, hyperparameter search, regulation-era handling check, and the NN readiness check.

Never promote more than one change per cycle, so effects stay attributable.

## 9. Autonomy boundaries

**You may change on your own:** features, model classes and hyperparameters (classical), calibration, simulation methods, data cleaning, dashboard pages, reports, tests, documentation, the backlog order.

**Protected. Never modify these in an autonomous cycle:**
- The **evaluation harness and metric definitions** (`eval/`), the **promotion gate**, and this file's sections 8 and 9
- `predictions/log.jsonl` (append-only) and any historical result data
- CI workflow permissions and secrets handling
- Anything that gets a model unfairly easier to pass the gate (including adjusting folds, thresholds, or baselines)

If an improvement seems to require touching something protected, **do not do it**. Write a proposal to `proposals/<date>-<slug>.md` explaining what and why, and surface it in the dashboard for a human.

Also never: force-push, commit secrets or API keys, add paid services, delete model versions, or rewrite git history.

**Logging is mandatory.** Every promotion, rejection, and rollback gets an entry in `CHANGELOG_MODELS.md` (date, what changed, metrics before/after, decision) and `experiments/log.jsonl`.

## 10. Streamlit dashboard

Reads from files committed in the repo (so Streamlit Community Cloud redeploys on every push). Pages:
1. **Next race**: win/podium/top-10 probabilities, predicted order, with the model version and timestamp
2. **Last weekend**: pace summary and prediction-vs-reality
3. **Championship**: title probabilities (drivers and constructors) over time
4. **Scorecard**: live record of predictions vs. baselines, calibration, confidence intervals
5. **Model log**: champion version, changelog, experiments, rollbacks, pending proposals, NN readiness

Always show uncertainty, never a bare "X will win."

## 11. Repo layout (target)

```
f1data.py, features.py, train.py, predict.py   # existing scaffold; refactor into src/ as it grows
src/            # ingestion, features, models, simulation
eval/           # PROTECTED: walk-forward harness, metrics, promotion gate
scripts/        # should_run.py gate, run_forecast.py, run_review.py
data/           # per-event CSVs, championship state
predictions/    # log.jsonl (append-only)
models/         # versioned registry, champion pointer
reports/        # weekly pace summaries
experiments/    # log.jsonl
proposals/      # items needing human decision
state/          # last_run.json (idempotency)
app/            # Streamlit dashboard
tests/          # leakage tests, pipeline tests, gate tests
.github/workflows/
CHANGELOG_MODELS.md
```

## 12. Milestones

- **M0** Verify real data pipeline, record baselines (update section 2)
- **M1** Persistent data store plus results ingestion, with tests
- **M2** Podium, top-10, and finishing-order models, plus DNF model
- **M3** Prediction log and scoring (Trigger A and B scripts working locally)
- **M4** Championship simulation
- **M5** Streamlit dashboard
- **M6** GitHub Actions scheduling, gate script, idempotency
- **M7** Model registry, promotion gate, shadow mode, rollback, changelog
- **M8** NN readiness checks (build NN only when readiness is met)

Work milestone by milestone. Don't start the automation (M6/M7) until M0 to M3 are verified on real data.

## 13. Working agreements

- Keep changes small, tested, and explained. Tests must pass before any merge.
- When uncertain about a protected or ambiguous matter, write a proposal instead of guessing.
- Be explicit about what is verified versus assumed. If the real-data run fails or numbers look too good, say so and investigate leakage first.
- Update section 2 whenever the project's state meaningfully changes.