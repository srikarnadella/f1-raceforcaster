# Proposal: enable practice data (needs F1TV credentials)

**Status:** needs a human decision. **Touches:** CI secrets handling (protected), a paid service (not allowed autonomously).

## What
Since FastF1 3.9, lap timing for every session (practice, qualifying laps, race laps with tyre compound) comes from
livetiming.formula1.com, which now requires an F1 account **with an active F1TV subscription**. Without it the
project runs on free Jolpica data only: qualifying Q1–Q3 times, sprint and race results, race lap times and pit stops.

## Why it matters
- Practice long-run pace and compound-aware stint analysis are the main signals the brief expects for "forecast from
  practice and qualifying". They are currently unavailable.
- `src/f1/practice.py` is written but **unverified**: it classifies practice runs (quali-sim / long-run / other) with
  k-means on run length, pace offset and tyre life, tracks compound, and fuel-corrects long runs.

## Options
1. Do nothing: keep the free-data pipeline (current state).
2. Add `F1_USERNAME` / `F1_PASSWORD` as GitHub Actions secrets from an existing F1TV subscription. Then verify
   `practice.py` on one weekend, backfill practice features, and run a normal gated experiment adding
   `fp_best_gap` and `fp_longrun_gap`.

Recommendation: option 2 only if a subscription already exists; the gain is unproven until it passes the gate.
