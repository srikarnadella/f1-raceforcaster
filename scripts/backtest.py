"""Walk-forward backtest of the champion (or a spec) against baselines.

Usage: python scripts/backtest.py [--spec models/specs/x.json] [--from 2019] [--out reports/backtest.json]
"""
import argparse
import json
import time

import _path  # noqa: F401
import numpy as np

from eval.harness import evaluate
from eval.metrics import bootstrap_ci
from f1 import ingest, registry
from f1.features import build_features
from f1.models import BASELINES, RaceModel


def run(spec, test_from, test_to=None, baselines=True, verbose=True):
    results = ingest.load_results()
    feats = build_features(results, **spec.get("form", {}))
    out = {}
    models = {"model": lambda: RaceModel(spec)}
    if baselines:
        models.update({k: v for k, v in BASELINES.items()})
    for name, factory in models.items():
        t = time.time()
        r = evaluate(factory, feats, test_from=test_from, test_to=test_to, verbose=verbose and name == "model")
        out[name] = r
        if verbose:
            print(f"{name:8s} {time.time()-t:5.0f}s  " + "  ".join(f"{k}={v:.4f}" for k, v in r["summary"].items()))
    return out


def report(out):
    rows = {}
    for name, r in out.items():
        s = dict(r["summary"])
        s.update(r["calibration"])
        s["n_races"] = len(r["per_race"])
        s["winner_logloss_ci"] = bootstrap_ci(r["per_race"], "winner_logloss")
        s["spearman_ci"] = bootstrap_ci(r["per_race"], "spearman")
        rows[name] = s
    return rows


def write(spec, path, test_from=2019, test_to=None, verbose=False, version=None):
    out = run(spec, test_from, test_to, verbose=verbose)
    by_season = {name: r["per_race"].groupby("season")[["winner_logloss", "podium_brier", "top10_brier", "spearman",
                                                          "top1_hit"]].mean().round(4).to_dict("index")
                 for name, r in out.items()}
    rep = report(out)
    json.dump({"model_version": version, "generated": registry.now(), "data_hash": registry.data_hash(),
               "overall": rep, "by_season": by_season}, open(path, "w"), indent=1, default=float)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--from", dest="test_from", type=int, default=2019)
    ap.add_argument("--to", dest="test_to", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    spec = json.load(open(a.spec)) if a.spec else registry.champion_spec()
    version = None if a.spec else registry.load()["champion"]
    rep = write(spec, a.out, a.test_from, a.test_to, verbose=True, version=version) if a.out else report(run(spec, a.test_from, a.test_to))
    print(json.dumps(rep, indent=1, default=float))
