"""Trigger B pieces: score logged predictions and write the weekend pace summary."""
from __future__ import annotations

import numpy as np
import pandas as pd

from eval.metrics import score_race
from . import laps as lapmod
from . import predlog
from .config import REPORTS
from .features import OUTCOME_COLS, build_features
from .forecast import history_before
from .models import BASELINES


def baseline_scores(results: pd.DataFrame, season: int, rnd: int) -> dict:
    race = results[(results.season == season) & (results["round"] == rnd)]
    feats = build_features(history_before(results, season, rnd), race.drop(columns=OUTCOME_COLS))
    train = feats[feats["position"].notna()]
    view = feats[(feats.season == season) & (feats["round"] == rnd)]
    out = {}
    for name, cls in BASELINES.items():
        out[name] = score_race(cls().fit(train).predict(view), race)
    return out


def score_logged(results: pd.DataFrame, season: int, rnd: int) -> list[dict]:
    race = results[(results.season == season) & (results["round"] == rnd)]
    if race.empty:
        return []
    base = baseline_scores(results, season, rnd)
    recs = []
    for e in predlog.find(season, rnd):
        m = score_race(predlog.entry_frame(e), race)
        rec = predlog.log_score(e, m, base)
        recs.append(rec or {"already_scored": True, **{k: e[k] for k in ("role", "model_version")}})
    return recs


def _hypotheses(row, race_info) -> list[str]:
    """Evidence-based, explicitly hedged explanations for a big miss."""
    h = []
    if not row["classified"]:
        h.append(f"did not finish (status: {row['status']})")
    if row["grid"] and row["grid"] - row["start_pos"] >= 3:
        h.append(f"started P{int(row['grid'])} vs qualified P{int(row['start_pos'])}: likely grid penalty (not modelled)")
    if row["grid"] == 0:
        h.append("pit-lane start")
    if race_info.get("neutral_laps", 0) and abs(row["miss"]) >= 4:
        h.append(f"hypothesis: {race_info['neutral_laps']} neutralised lap(s) (SC/VSC) may have reshuffled strategy")
    if not np.isnan(row.get("stops", np.nan)) and race_info.get("median_stops") is not None \
            and row["stops"] != race_info["median_stops"]:
        h.append(f"hypothesis: different strategy ({int(row['stops'])} stops vs field median {race_info['median_stops']:.0f})")
    if not np.isnan(row.get("race_pace_rank", np.nan)) and abs(row["race_pace_rank"] - row["start_pos"]) >= 5:
        h.append(f"race pace rank P{int(row['race_pace_rank'])} vs qualified P{int(row['start_pos'])}: one-lap and race pace disagreed")
    return h or ["no clear evidence in available data (hypothesis: race incident or strategy)"]


def pace_report(results: pd.DataFrame, season: int, rnd: int) -> str:
    race = results[(results.season == season) & (results["round"] == rnd)].copy()
    name = race["race_name"].iloc[0]
    feats = build_features(history_before(results, season, rnd), race.drop(columns=OUTCOME_COLS))
    wk = feats[(feats.season == season) & (feats["round"] == rnd)][["driver_id", "start_pos", "quali_gap"]]
    race = race.merge(wk, on="driver_id", how="left")

    try:
        rp = lapmod.driver_race_pace(lapmod.fetch_race_laps(season, rnd))
        race_info = {"neutral_laps": rp.attrs.get("neutral_laps", 0), "median_stops": float(rp["stops"].median()),
                     "fuel": rp.attrs.get("fuel_s_per_lap"), "deg": rp.attrs.get("deg_s_per_lap")}
        race = race.merge(rp, on="driver_id", how="left")
        race["race_pace_rank"] = race["race_pace_gap"].rank(method="min")
        compounds = lapmod.infer_compounds(lapmod.fetch_race_laps(season, rnd))
    except Exception as ex:  # lap data can lag; the report still renders
        race_info, compounds = {"error": str(ex)}, None
        for c in ["race_pace_gap", "race_pace_rank", "stops", "n_clean", "n_traffic"]:
            race[c] = np.nan

    L = [f"# {season} R{rnd}: {name}, weekend pace summary", ""]
    L += ["Pace below uses qualifying and race laps (Jolpica); practice long runs, real tyre compounds and "
          "telemetry for this weekend are in the dashboard's Weekend lab and Race story (OpenF1).", ""]
    # structured copy of everything below, for the dashboard (reports/<season>_<round>_pace.json)
    J: dict = {"season": season, "round": rnd, "race_name": name}

    # ---- team table
    team = race.groupby("constructor_id").agg(quali_gap=("quali_gap", "min"), race_pace_gap=("race_pace_gap", "min"),
                                              best_finish=("position", "min"), points=("points", "sum"))
    team["race_minus_quali"] = team["race_pace_gap"] - team["quali_gap"]
    L += ["## Team pace (best driver, % gap to fastest)", "",
          "| Team | Quali gap % | Race pace gap % (fuel/tyre-corrected) | Race − quali | Best finish | Points |",
          "|---|---|---|---|---|---|"]
    for c, r in team.sort_values("quali_gap").iterrows():
        L.append(f"| {c} | {r.quali_gap:.3f} | {_f(r.race_pace_gap)} | {_f(r.race_minus_quali)} | P{int(r.best_finish)} | {r.points:g} |")
    J["teams"] = team.reset_index().rename(columns={"constructor_id": "team"}).round(4).to_dict("records")
    J["race_model"] = {k: race_info.get(k) for k in ("fuel", "deg", "neutral_laps", "median_stops")}
    if "fuel" in race_info and race_info["fuel"] is not None:
        L += ["", f"Race lap model: fuel effect {race_info['fuel']:+.3f} s/lap, tyre degradation {race_info['deg']:+.3f} s/lap "
              f"(field-wide fit on clean and traffic laps); {race_info['neutral_laps']} neutralised lap(s) detected."]

    # ---- driver table and prediction review
    entries = [e for e in predlog.find(season, rnd, role="champion")]
    pred = predlog.entry_frame(entries[-1]) if entries else None
    if pred is not None:
        race = race.merge(pred[["driver_id", "p_win", "p_top3", "exp_pos"]], on="driver_id", how="left")
    else:
        race["p_win"] = race["p_top3"] = race["exp_pos"] = np.nan
    race["miss"] = race["position"] - race["exp_pos"]

    L += ["", "## Drivers", "",
          "| Driver | Team | Quali | Quali gap % | Race pace gap % | Clean / traffic laps | Stops | Predicted pos | Finish | Status |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in race.sort_values("position").iterrows():
        L.append(f"| {r.code} | {r.constructor_id} | P{int(r.start_pos)} | {r.quali_gap:.3f} | {_f(r.race_pace_gap)} | "
                 f"{_i(r.n_clean)} / {_i(r.n_traffic)} | {_i(r.stops)} | {_f(r.exp_pos, 1)} | P{int(r.position)} | {r.status} |")

    J["drivers"] = race.sort_values("position")[["code", "constructor_id", "start_pos", "quali_gap", "race_pace_gap",
                                                    "n_clean", "n_traffic", "stops", "exp_pos", "p_win", "position",
                                                    "status"]].round(4).to_dict("records")
    fin = race[race["classified"].astype(bool)]
    if pred is not None and len(fin):
        over = fin.nsmallest(3, "miss")
        under = race.nlargest(3, "miss")
        J["over"] = [{"code": r.code, "expected": round(r.exp_pos, 1), "finish": int(r.position)} for _, r in over.iterrows()]
        J["under"] = [{"code": r.code, "expected": round(r.exp_pos, 1), "finish": int(r.position)} for _, r in under.iterrows()]
        L += ["", "## Over / under-delivered vs forecast", "",
              "Over-delivered: " + ", ".join(f"{r.code} (exp {r.exp_pos:.1f} → P{int(r.position)})" for _, r in over.iterrows()),
              "", "Under-delivered: " + ", ".join(f"{r.code} (exp {r.exp_pos:.1f} → P{int(r.position)})" for _, r in under.iterrows())]

    L += ["", "## Prediction review", ""]
    sc = []
    if pred is None:
        L.append("No pre-race forecast was logged for this race, so there is nothing to score.")
    else:
        e = entries[-1]
        winner = race.loc[race["position"] == 1].iloc[0]
        top = pred.sort_values("p_win", ascending=False).head(5)
        L.append(f"Forecast by model {e['model_version']} logged {e['logged_at']} (race start {e['race_utc']})"
                 + (" — **LATE, not counted**" if e["late"] else "") + ".")
        L.append("")
        L.append("Top win probabilities: " + ", ".join(f"{r.code} {r.p_win:.0%}" for _, r in top.iterrows())
                 + f". Winner: **{winner.code}** (forecast {winner.p_win:.0%}).")
        J["forecast"] = {"model_version": e["model_version"], "logged_at": e["logged_at"], "race_utc": e["race_utc"],
                         "late": e["late"], "top": [{"code": r.code, "p_win": float(r.p_win)} for _, r in top.iterrows()],
                         "winner": winner.code, "winner_p": float(winner.p_win)}
        sc = [s for s in predlog.read_scores() if s["season"] == season and s["round"] == rnd and s["role"] == "champion"]
        if sc:
            m, b = sc[-1]["metrics"], sc[-1]["baselines"]
            L += ["", "| Metric | Model | Grid baseline | Pole baseline | Uniform |", "|---|---|---|---|---|"]
            for k in ["winner_logloss", "podium_brier", "top10_brier", "spearman", "mae_pos"]:
                L.append(f"| {k} | {m[k]:.3f} | {b['grid'][k]:.3f} | {b['pole'][k]:.3f} | {b['uniform'][k]:.3f} |")
            J["scores"] = {k: {"model": m[k], "grid": b["grid"][k], "pole": b["pole"][k], "uniform": b["uniform"][k]}
                           for k in ["winner_logloss", "podium_brier", "top10_brier", "spearman", "mae_pos"]}
        L += ["", "Biggest misses (explanations are hypotheses unless stated as recorded status):", ""]
        J["misses"] = []
        for _, r in race.reindex(race["miss"].abs().sort_values(ascending=False).index).head(5).iterrows():
            L.append(f"- **{r.code}**: expected {r.exp_pos:.1f}, finished P{int(r.position)}. " + "; ".join(_hypotheses(r, race_info)))
            J["misses"].append({"code": r.code, "expected": round(float(r.exp_pos), 1), "finish": int(r.position),
                                "hypotheses": _hypotheses(r, race_info)})

    # ---- takeaways
    T = []
    if (race["grid"] - race["start_pos"] >= 3).any():
        T.append("Grid penalties changed the start order; the model uses qualifying position (penalties not yet a feature).")
    if pred is not None:
        dnf_top = race[(~race["classified"].astype(bool)) & (race["p_top3"] > 0.2)]
        if len(dnf_top):
            T.append("DNF of a podium contender: " + ", ".join(dnf_top["code"]) + "; check the reliability model.")
    big = team[(team["race_minus_quali"].abs() > 0.4)]
    for c, r in big.iterrows():
        T.append(f"{c}: race pace {'worse' if r.race_minus_quali > 0 else 'better'} than one-lap pace by {abs(r.race_minus_quali):.2f}% "
                 "(supports a race-vs-quali pace feature).")
    if compounds is not None and len(compounds) and "inferred_class" in compounds:
        mix = compounds["inferred_class"].value_counts().to_dict()
        T.append(f"Inferred stint classes (unsupervised, not observed compounds): {mix}.")
    if sc and m["winner_logloss"] > b["grid"]["winner_logloss"]:
        T.append("Winner log-loss worse than the grid baseline this race.")
    L += ["", "## Takeaways for the model", ""] + [f"- {t}" for t in (T or ["Nothing unusual flagged."])]
    text = "\n".join(L) + "\n"
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{season}_{rnd:02d}_pace.md").write_text(text)  # local, gitignored
    J["takeaways"] = T or ["Nothing unusual flagged."]
    import json
    (REPORTS / f"{season}_{rnd:02d}_pace.json").write_text(json.dumps(J, indent=1, default=_jsonable))
    return text


def _jsonable(o):
    if isinstance(o, np.floating | np.integer):
        return None if isinstance(o, np.floating) and np.isnan(o) else o.item()
    if isinstance(o, float) and np.isnan(o):
        return None
    return str(o)


def _f(x, d=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _i(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{int(x)}"
