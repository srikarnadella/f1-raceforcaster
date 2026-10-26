"""P1 / podium lab: candidates aimed at the owner's objective (winner first,
podium second), each run through the promotion gate against the champion.

Reports the frozen 2019+ gate window plus 2025-2026 (the only window that is
out-of-sample for the simulator, whose parameters were tuned on 2023-2024)
and 2026 alone. Never promotes: the weekly cycle does that.

Usage: python scripts/podium_lab.py
"""
import json
import warnings


from eval import gate
from eval.metrics import summarize
from f1 import improve, ingest, registry, tracking
from f1.config import REPORTS

KEYS = ["winner_logloss", "winner_brier", "top1_hit", "podium_brier", "top10_brier", "spearman"]


def main():
    warnings.simplefilter("ignore")
    res = ingest.load_results()
    reg = registry.ensure_initialized()
    champ = registry.champion_spec(reg)
    runs = json.loads((REPORTS / "race_sim_lab.json").read_text())["runs"]
    sim = runs[max(runs, key=lambda k: k.split("-")[-1])]["params"]  # latest calibration
    sim = {k: v for k, v in sim.items() if k != "n_sims"} | {"n_sims": 1000}
    cands = {
        "win-temperature-1.2": {"strength": {"win_temperature": 1.2}},
        "win-temperature-1.3": {"strength": {"win_temperature": 1.3}},
        "win-temperature-1.4": {"strength": {"win_temperature": 1.4}},
        "depth-1 (train on winners only)": {"strength": {"depth": 1}},
        "depth-2": {"strength": {"depth": 2}},
        "simulator blend 30%": {"simulator": {**sim, "weight": 0.3}},
        "simulator blend 50%": {"simulator": {**sim, "weight": 0.5}},
    }
    ce = improve._eval_cached(champ, res)
    out = {"objective": "winner log-loss 60% + podium Brier 40% (owner, 2026-10-25)", "champion": reg["champion"],
           "champion_metrics": {}, "candidates": {}}

    def windows(e):
        pr = e["per_race"]
        return {"2019-2026": {k: float(summarize(pr)[k]) for k in KEYS},
                "2025-2026": {k: float(summarize(pr[pr.season >= 2025])[k]) for k in KEYS},
                "2026": {k: float(summarize(pr[pr.season == 2026])[k]) for k in KEYS}}
    out["champion_metrics"] = windows(ce)
    for name, patch in cands.items():
        spec = improve.deep_merge(champ, patch)
        e = improve._eval_cached(spec, res)
        d = gate.decide(e, ce, run_leakage=False)
        pr, pc = e["per_race"], ce["per_race"]
        late = gate.composite(summarize(pr[pr.season >= 2025]), summarize(pc[pc.season >= 2025]))[0]
        tracking.log_evaluation("podium-lab", name, spec, e, d, tags={"champion": reg["champion"],
                                                                  "composite_2025_2026": round(late, 4)})
        out["candidates"][name] = {"patch": patch, "gate_pass": d["promote"], "checks": d["checks"],
                                   "composite": d["composite"], "relative": d["relative"],
                                   "bootstrap": d["bootstrap_win_rate"], "composite_2025_2026": late,
                                   "metrics": windows(e)}
        print(f"{name:32s} composite {d['composite']:+.4f}  2025-26 {late:+.4f}  boot {d['bootstrap_win_rate']:.2f}  "
              f"pass {d['promote']}  {[k for k, v in d['checks'].items() if not v]}", flush=True)
    (REPORTS / "podium_lab.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
