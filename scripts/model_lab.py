"""Model lab: walk-forward comparison of race-winner models on the frozen harness.

  grid, pole            baselines
  champion              current champion (see models/registry.json)
  v1+practice           + OpenF1 pre-race features (practice long runs and one-lap
                        pace, sprint pace, quali ideal lap, telemetry DNA projection)
  nn                    neural listwise ranker on the champion's features
  nn+practice           neural ranker + OpenF1 features

Neural hyperparameters are fixed in advance (not tuned on the test races).
Windows: 2023 onward (OpenF1 era) and 2026 alone. Candidates are also run
through the promotion gate on the full frozen 2019+ window; this script
reports the gate result but never promotes (that stays with the weekly cycle).

Usage: python scripts/model_lab.py [--gate]
"""
import argparse
import copy
import json
import warnings

import pandas as pd

from eval import gate
from eval.harness import evaluate
from eval.metrics import summarize
from f1 import ingest, registry, tracking
from f1.analytics.prerace import OF1_FEATURES
from f1.config import REPORTS
from f1.features import build_features
from f1.models import BASELINES, RaceModel

COLS = ["n_races", "winner_logloss", "podium_brier", "top10_brier", "spearman", "top1_hit", "mae_pos"]


def specs(champ: dict) -> dict:
    out = {"champion": champ}
    pr = copy.deepcopy(champ)
    pr["name"] = "pl-dnf+practice"
    pr["strength"]["features"] = champ["strength"]["features"] + OF1_FEATURES
    pr["form"] = {**champ.get("form", {}), "openf1": True}
    out["v1+practice"] = pr
    nn = copy.deepcopy(champ)
    nn["name"] = "nn-pl"
    nn["strength"].update(hidden=8, l2_hidden=4.0, n_models=3)
    out["nn"] = nn
    nnp = copy.deepcopy(pr)
    nnp["name"] = "nn-pl+practice"
    nnp["strength"].update(hidden=8, l2_hidden=4.0, n_models=3)
    out["nn+practice"] = nnp
    try:
        import torch  # noqa: F401
        nt = copy.deepcopy(nn)
        nt["name"] = "nn-pl-torch"
        nt["strength"]["backend"] = "torch"
        out["nn (pytorch)"] = nt
    except ImportError:
        pass
    return out


def table(per_race: pd.DataFrame) -> dict:
    s = summarize(per_race)
    return {k: float(s[k]) for k in COLS if k in s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="also run the promotion gate on the 2019+ window")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    reg = registry.ensure_initialized()
    champ = reg["versions"][reg["champion"]]["spec"]
    results = ingest.load_results()
    feats = build_features(results, **{**champ.get("form", {}), "openf1": True})
    models = {"grid": BASELINES["grid"], "pole": BASELINES["pole"]}
    sp = specs(champ)
    evals = {}
    for name, f in models.items():
        evals[name] = evaluate(f, feats, test_from=2023)
        print(name, "done", flush=True)
    for name, s in sp.items():
        evals[name] = evaluate(lambda s=s: RaceModel(s), feats, test_from=2023)
        tracking.log_evaluation("model-lab", name, s, evals[name], tags={"window": "2023+"})
        print(name, table(evals[name]["per_race"]), flush=True)
    windows = {"2023-2026": {}, "2026": {}}
    for name, e in evals.items():
        pr = e["per_race"]
        windows["2023-2026"][name] = table(pr)
        windows["2026"][name] = table(pr[pr["season"] == 2026])
    out = {"note": "Walk-forward on the frozen harness: each race predicted from earlier races only. "
                   "Lower log-loss / Brier is better; higher Spearman / top-1 hit is better. "
                   "~20 races per season, so differences under ~3% are within noise.",
           "champion": reg["champion"], "windows": windows, "specs": {k: v for k, v in sp.items()}}
    if a.gate:
        full_feats = feats
        champ_full = evaluate(lambda: RaceModel(champ), full_feats)
        out["gate"] = {}
        for name in ["v1+practice", "nn", "nn+practice"]:
            cand = evaluate(lambda s=sp[name]: RaceModel(s), full_feats)
            d = gate.decide(cand, champ_full)
            out["gate"][name] = {"promote": d["promote"], "checks": d["checks"], "composite": round(d["composite"], 4),
                                 "relative": {k: round(v, 4) for k, v in d["relative"].items()},
                                 "bootstrap_win_rate": d["bootstrap_win_rate"]}
            print("gate", name, out["gate"][name], flush=True)
    (REPORTS / "model_lab.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(windows, indent=1))


if __name__ == "__main__":
    main()
