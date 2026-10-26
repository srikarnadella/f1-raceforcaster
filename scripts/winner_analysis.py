"""Why does the champion miss race winners? Walk-forward predictions of the
current champion (frozen harness, 2019 onward) broken down by cause.

Writes reports/winner_analysis.json. Usage: python scripts/winner_analysis.py
"""
import json
import warnings

import numpy as np
import pandas as pd

from eval.harness import walk_forward
from f1 import ingest, registry
from f1.config import LAPS_DIR, REPORTS, era_of
from f1.features import build_features
from f1.models import RaceModel


def neutral_laps(season, rnd):
    """SC/VSC laps from stored race laps (field median > 115% of race median)."""
    p = LAPS_DIR / f"{season}_{rnd:02d}.csv"
    if not p.exists():
        return np.nan
    from f1.laps import categorise
    l = categorise(pd.read_csv(p))
    return int((l.groupby("lap")["category"].apply(lambda s: (s == "neutral").mean() > 0.5)).sum())


def main():
    warnings.simplefilter("ignore")
    reg = registry.ensure_initialized()
    spec = registry.champion_spec(reg)
    res = ingest.load_results()
    feats = build_features(res, **spec.get("form", {}))
    per_race, preds = walk_forward(lambda: RaceModel(spec), feats)
    wk = res.drop_duplicates(["season", "round"]).set_index(["season", "round"])[["race_name", "circuit_id"]]
    rows = []
    for (s, r), g in preds.groupby(["season", "round"]):
        g = g.sort_values("p_win", ascending=False).reset_index(drop=True)
        actual = res[(res.season == s) & (res["round"] == r)].set_index("driver_id")
        win = g[g["position"] == 1]
        if win.empty:
            continue
        w = win.iloc[0]
        fav = g.iloc[0]
        fa = actual.loc[fav.driver_id] if fav.driver_id in actual.index else None
        rows.append({
            "season": int(s), "round": int(r), "race": wk.loc[(s, r), "race_name"], "era": era_of(int(s)),
            "winner": w.code, "winner_start": int(w.start_pos), "winner_p": float(w.p_win),
            "winner_rank": int(win.index[0]) + 1,
            "fav": fav.code, "fav_p": float(fav.p_win), "fav_start": int(fav.start_pos),
            "fav_dnf": bool(fa is not None and not bool(fa["classified"])),
            "same_team": bool(fav.constructor_id == w.constructor_id),
            "hit": bool(fav.driver_id == w.driver_id),
            "neutral_laps": neutral_laps(int(s), int(r)),
        })
    A = pd.DataFrame(rows)
    misses = A[~A["hit"]]

    def cause(m):
        if m.fav_dnf:
            return "favourite did not finish"
        if m.same_team:
            return "favourite's teammate won"
        if m.winner_start >= 4:
            return "winner started P4 or lower"
        if pd.notna(m.neutral_laps) and m.neutral_laps >= 3:
            return "safety-car race reshuffled it"
        return "another front-runner was faster on the day"
    causes = misses.apply(cause, axis=1).value_counts()
    # calibration of the favourite's probability
    A["fav_bucket"] = pd.cut(A["fav_p"], [0, .3, .45, .6, .75, 1.0])
    cal = A.groupby("fav_bucket", observed=True).agg(races=("hit", "size"), predicted=("fav_p", "mean"),
                                                     actual=("hit", "mean")).reset_index()
    cal["fav_bucket"] = cal["fav_bucket"].astype(str)
    by_start = A.groupby(A["winner_start"].clip(upper=6))["winner_p"].agg(["size", "mean"]).reset_index()
    by_start.columns = ["winner_start (6 = 6+)", "races", "mean_p_given_to_winner"]
    by_season = A.groupby("season").agg(races=("hit", "size"), hit_rate=("hit", "mean"),
                                        winner_p=("winner_p", "mean"), fav_p=("fav_p", "mean")).reset_index()
    out = {
        "races": int(len(A)), "hit_rate": float(A["hit"].mean()), "mean_p_winner": float(A["winner_p"].mean()),
        "mean_fav_p": float(A["fav_p"].mean()),
        "miss_causes": {k: int(v) for k, v in causes.items()},
        "calibration": cal.to_dict("records"), "by_winner_start": by_start.to_dict("records"),
        "by_season": by_season.to_dict("records"),
        "worst_misses": A.sort_values("winner_p").head(12).to_dict("records"),
        "notes": [],
    }
    # headline observations computed from the data, not written by hand
    o = out["notes"]
    o.append(f"The favourite won {out['hit_rate']:.0%} of {len(A)} races while being given {out['mean_fav_p']:.0%} on "
             f"average: {'under' if out['hit_rate'] > out['mean_fav_p'] + 0.03 else 'over' if out['hit_rate'] < out['mean_fav_p'] - 0.03 else 'well'}-confident.")
    top = causes.index[0] if len(causes) else None
    if top:
        o.append(f"Most common miss: {top} ({causes.iloc[0]} of {len(misses)} misses).")
    (REPORTS / "winner_analysis.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("worst_misses",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
