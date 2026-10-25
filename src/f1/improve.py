"""Improvement cycle (CLAUDE.md section 8): at most 3 experiments and 30 minutes
per cycle, evaluated on the frozen walk-forward harness, gated by eval/gate.py.
At most one promotion per cycle. Every outcome is logged.

Candidates come from experiments/queue.json: [{"slug", "hypothesis", "patch"}],
where `patch` is deep-merged into the champion spec (one change at a time).
Code-changing experiments are made by a human or Claude session on an
exp/<date>-<slug> branch and then queued here as a spec referencing the new code.
"""
from __future__ import annotations

import copy
import json
import pickle
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from eval import gate
from eval.harness import evaluate
from eval.metrics import COMPOSITE
from . import ingest, predlog, registry
from .config import CACHE, CHANGELOG, EXP_LOG, EXP_QUEUE, PROPOSALS
from .features import build_features
from .models import RaceModel

MAX_EXPERIMENTS = 3
BUDGET_S = 30 * 60
NN_RACES_TARGET = 150


def deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _eval_cached(spec: dict, results: pd.DataFrame) -> dict:
    h = registry.data_hash() + json.dumps(spec, sort_keys=True)
    import hashlib
    p = CACHE / "evals" / (hashlib.sha1(h.encode()).hexdigest() + ".pkl")
    if p.exists():
        return pickle.loads(p.read_bytes())
    feats = build_features(results, **spec.get("form", {}))
    r = evaluate(lambda: RaceModel(spec), feats)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(r))
    return r


def read_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def append_jsonl(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=float, sort_keys=True) + "\n")


def changelog(lines: list[str]):
    if not CHANGELOG.exists():
        CHANGELOG.write_text("# Model changelog\n\nEvery promotion, rejection and rollback. Newest last.\n")
    with open(CHANGELOG, "a") as f:
        f.write("\n" + "\n".join(lines) + "\n")


def nn_readiness(results: pd.DataFrame, reg: dict) -> dict:
    n_races = int(results[["season", "round"]].drop_duplicates().shape[0])
    promos = [h for h in reg["history"] if h["event"] == "promote"][-3:]
    gains = [h.get("composite") for h in promos if h.get("composite") is not None]
    plateau = len(gains) == 3 and all(g < 0.01 for g in gains)
    design = False  # flips only when a proposal with a concrete NN design exists
    ready = n_races >= NN_RACES_TARGET and plateau and design
    line = (f"NN readiness: races={n_races} (target {NN_RACES_TARGET}: {'met' if n_races >= NN_RACES_TARGET else 'not met'}); "
            f"plateau={'yes' if plateau else 'no'} (last promotions {gains or 'none'}); "
            f"design rationale={'yes' if design else 'no'} -> {'READY: propose challenger' if ready else 'not ready'}")
    return {"ready": ready, "races": n_races, "plateau": plateau, "design": design, "line": line}


# One-change neighbourhood around the champion, so the cycle never runs dry.
NUMERIC_MOVES = [
    ("strength.l2", lambda v: [v * 0.3, v * 3]),
    ("strength.depth", lambda v: [max(1, v - 1), v + 1]),
    ("strength.season_half_life", lambda v: [v * 0.5, v * 2]),
    ("dnf.C", lambda v: [v * 0.3, v * 3]),
    ("dnf.season_half_life", lambda v: [v * 0.5, v * 2]),
    ("form.form_half_life", lambda v: [v * 0.67, v * 1.5]),
    ("form.dnf_half_life", lambda v: [v * 0.5, v * 2]),
]
RETRY_AFTER_RACES = 6   # a rejected change becomes eligible again once this much new data exists


def _get(spec, path, default=None):
    cur = spec
    for k in path.split("."):
        cur = cur.get(k, {}) if isinstance(cur, dict) else {}
    return cur if cur != {} else default


def _patch(path, value) -> dict:
    out = value
    for k in reversed(path.split(".")):
        out = {k: out}
    return out


def generate_candidates(spec: dict) -> list[dict]:
    from .features import RACE_PACE_FEATURES
    defaults = {"form.form_half_life": 6.0, "form.dnf_half_life": 12.0}
    cands = []
    for path, moves in NUMERIC_MOVES:
        v = _get(spec, path, defaults.get(path))
        if v is None:
            continue
        for nv in moves(v):
            nv = int(nv) if isinstance(v, int) else round(float(nv), 4)
            if nv != v:
                cands.append({"slug": f"{path}={nv}", "hypothesis": f"auto: {path} {v} -> {nv}",
                              "patch": _patch(path, nv)})
    feats = spec["strength"]["features"]
    for f in feats:
        cands.append({"slug": f"drop:{f}", "hypothesis": f"auto: ablate strength feature {f}",
                      "patch": {"strength": {"features": [x for x in feats if x != f]}}})
    if not set(RACE_PACE_FEATURES) & set(feats):
        cands.append({"slug": "add:race_pace", "hypothesis": "auto: add lap-derived race-pace features",
                      "patch": {"form": {"race_pace": True}, "strength": {"features": feats + RACE_PACE_FEATURES}}})
    return cands


def _attempt_key(slug: str, champion: str, n_races: int) -> str:
    return f"{slug}|{champion}|{n_races // RETRY_AFTER_RACES}"


def pick_candidates(champ_v: str, champ_spec: dict, n_races: int, log: list[dict]) -> list[dict]:
    """Queued (human/Claude) hypotheses first, then auto-generated ones in a stable
    rotating order. Generated candidates already tried against this champion are
    skipped until RETRY_AFTER_RACES more races of data exist."""
    import hashlib
    done_slugs = {e["slug"] for e in log}
    tried = {e.get("attempt_key") for e in log}
    queue = [q for q in (json.loads(EXP_QUEUE.read_text()) if EXP_QUEUE.exists() else []) if q["slug"] not in done_slugs]
    gen = [c for c in generate_candidates(champ_spec) if _attempt_key(c["slug"], champ_v, n_races) not in tried]
    gen.sort(key=lambda c: hashlib.sha1(f"{c['slug']}|{champ_v}".encode()).hexdigest())
    for c in gen:
        c["auto"] = True
    return queue + gen


def run_cycle(next_race: tuple[int, int], dry_run: bool = False) -> dict:
    t0 = time.time()
    results = ingest.load_results()
    reg = registry.ensure_initialized()
    champ_v = reg["champion"]
    champ_spec = registry.champion_spec(reg)
    n_races = int(results[["season", "round"]].drop_duplicates().shape[0])
    queue = pick_candidates(champ_v, champ_spec, n_races, read_jsonl(EXP_LOG))
    champ = _eval_cached(champ_spec, results)
    tried, best = [], None
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for q in queue[:MAX_EXPERIMENTS]:
        if time.time() - t0 > BUDGET_S:
            break
        spec = deep_merge(champ_spec, q["patch"])
        spec["name"] = q["slug"]
        try:
            cand = _eval_cached(spec, results)
            d = gate.decide(cand, champ)
        except Exception as ex:
            d, cand = {"promote": False, "error": repr(ex)}, None
        rec = {"date": date, "slug": q["slug"], "attempt_key": _attempt_key(q["slug"], champ_v, n_races),
               "auto": bool(q.get("auto")), "branch": f"exp/{date}-{q['slug']}", "hypothesis": q.get("hypothesis"),
               "patch": q["patch"], "champion": champ_v, "decision": d,
               "champion_metrics": champ["summary"], "candidate_metrics": cand["summary"] if cand else None}
        tried.append((rec, spec))
        if d.get("promote") and (best is None or d["composite"] > best[0]["decision"]["composite"]):
            best = (rec, spec)
    promoted = None
    if best and not dry_run:
        rec, spec = best
        v = registry.next_version(reg)
        registry.register(reg, v, spec, note=rec["hypothesis"] or rec["slug"], parent=champ_v,
                          backtest=rec["candidate_metrics"])
        registry.promote(reg, v, next_race, reason=rec["slug"])
        reg["history"][-1]["composite"] = rec["decision"]["composite"]
        registry.save(reg)
        promoted = v
    for rec, spec in tried:
        rec["outcome"] = ("promoted as " + promoted) if best and rec is best[0] and promoted else (
            "passed gate, not chosen (one promotion per cycle)" if rec["decision"].get("promote") else "rejected")
        if dry_run:
            continue
        append_jsonl(EXP_LOG, rec)
        d = rec["decision"]
        lines = [f"## {rec['date']} — {rec['slug']}: {rec['outcome']}",
                 f"- Change: `{json.dumps(rec['patch'])}` vs champion {champ_v}",
                 f"- Hypothesis: {rec['hypothesis']}"]
        if "error" in d:
            lines.append(f"- Error: {d['error']}")
        else:
            lines.append(f"- Composite {d['composite']:+.2%}, bootstrap win rate {d['bootstrap_win_rate']:.0%}, "
                         f"checks {d['checks']}")
            lines.append("- Before → after: " + ", ".join(
                f"{m} {rec['champion_metrics'][m]:.4f} → {rec['candidate_metrics'][m]:.4f}" for m in COMPOSITE))
        changelog(lines)
    nn = nn_readiness(results, reg)
    summary = {"date": date, "event": "cycle", "champion_before": champ_v, "promoted": promoted,
               "experiments": [r["slug"] for r, _ in tried], "nn_readiness": nn["line"],
               "seconds": round(time.time() - t0, 1)}
    if not dry_run:
        append_jsonl(EXP_LOG, {**summary, "slug": f"_cycle_{date}_{len(read_jsonl(EXP_LOG))}"})
    return summary


def check_rollback(reg: dict) -> str | None:
    """Live shadow comparison. Returns the rolled-back version or None."""
    if not reg.get("shadow"):
        return None
    s0, r0 = reg["shadow_since"]
    scores = [s for s in predlog.read_scores() if not s["late"] and (s["season"], s["round"]) >= (s0, r0)]
    champ = {(s["season"], s["round"]): s["metrics"] for s in scores if s["model_version"] == reg["champion"]}
    shadow = {(s["season"], s["round"]): s["metrics"] for s in scores if s["model_version"] == reg["shadow"]}
    both = sorted(set(champ) & set(shadow))
    if not both:
        return None
    cm = {m: np.mean([champ[k][m] for k in both]) for m in COMPOSITE}
    sm = {m: np.mean([shadow[k][m] for k in both]) for m in COMPOSITE}
    comp, _ = gate.composite(cm, sm)
    if comp < -registry.ROLLBACK_MARGIN:
        bad = registry.rollback(reg, f"live composite {comp:+.2%} vs shadow over {len(both)} races")
        registry.save(reg)
        changelog([f"## {datetime.now(timezone.utc):%Y-%m-%d} — ROLLBACK {bad} → {reg['champion']}",
                   f"- Live composite {comp:+.2%} vs previous champion over {len(both)} race(s) (threshold −2%)."])
        append_jsonl(EXP_LOG, {"slug": f"_rollback_{bad}_{len(both)}", "event": "rollback", "from": bad,
                               "to": reg["champion"], "composite": comp, "races": both})
        return bad
    if len(both) >= registry.SHADOW_RACES:
        reg["history"].append({"date": registry.now(), "event": "shadow_retired", "version": reg["shadow"],
                               "composite": comp})
        reg["shadow"], reg["shadow_since"] = None, None
        registry.save(reg)
    return None


def write_proposal(slug: str, body: str):
    PROPOSALS.mkdir(exist_ok=True)
    p = PROPOSALS / f"{datetime.now(timezone.utc):%Y-%m-%d}-{slug}.md"
    if not p.exists():
        p.write_text(body)
    return p
