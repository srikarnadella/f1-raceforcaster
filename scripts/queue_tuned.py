"""After tuning, propose the newly calibrated race-simulator settings as an experiment.

Reads the latest Optuna calibration (reports/race_sim_lab.json) and, if its
settings differ from the champion's simulator, puts a candidate at the front of
experiments/queue.json. It never promotes: the next post-race improvement cycle
runs it through the promotion gate like any other candidate.

Usage: python scripts/queue_tuned.py [--calib 2023-2024]
"""
import argparse
import json
from datetime import UTC, datetime

from f1 import registry
from f1.config import EXP_QUEUE, REPORTS

TUNED = ["w_quali", "pace_unc", "c1", "c2", "start_noise", "lap_noise"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=None, help="which calibration run to use (default: the latest)")
    a = ap.parse_args()
    runs = json.loads((REPORTS / "race_sim_lab.json").read_text())["runs"]
    key = a.calib or max(runs, key=lambda k: k.split("-")[-1])
    params = runs[key]["params"]
    champ = registry.champion_spec(registry.ensure_initialized())
    current = champ.get("simulator", {})
    if current and all(abs(float(current.get(k, -1)) - float(params[k])) < 1e-6 for k in TUNED):
        print("champion already uses these simulator settings; nothing queued")
        return
    sim = {**current, **{k: params[k] for k in TUNED}, "weight": current.get("weight", 0.5), "n_sims": 1000}
    slug = f"simulator-tuned-{datetime.now(UTC):%Y-%m-%d}"
    queue = json.loads(EXP_QUEUE.read_text()) if EXP_QUEUE.exists() else []
    queue = [q for q in queue if not q["slug"].startswith("simulator-tuned-")]  # newest tuning replaces older ones
    queue.insert(0, {"slug": slug, "patch": {"simulator": sim},
                     "hypothesis": f"Scheduled Optuna re-calibration of the race simulator on {key} "
                                   f"(reports/race_sim_lab.json); settings {json.dumps({k: round(params[k], 3) for k in TUNED})}."})
    EXP_QUEUE.write_text(json.dumps(queue, indent=1))
    print("queued", slug)


if __name__ == "__main__":
    main()
