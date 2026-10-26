"""Experiment tracking with MLflow (local SQLite store ./mlflow.db, gitignored).

Every model evaluation that matters is logged as a run: the improvement cycle's
candidates (with the gate decision), the podium / model / simulator labs and the
Optuna calibration trials. Browse with:  mlflow ui --backend-store-uri sqlite:///mlflow.db

Tracking is best effort: if MLflow is missing or F1_TRACKING=0, every call is a
no-op, so the pipeline and CI never depend on it.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager

from .config import ROOT

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"


def _mlflow():
    if os.environ.get("F1_TRACKING", "1") == "0":
        return None
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, f"{key}."))
        else:
            out[key] = v
    return out


@contextmanager
def run(experiment: str, name: str, params: dict | None = None, tags: dict | None = None, nested: bool = False):
    """with tracking.run("improvement-cycle", "win-temperature-1.3", params=spec) as log: log(metrics)"""
    mlflow = _mlflow()
    if mlflow is None:
        yield lambda *a, **k: None
        return
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name, nested=nested):
        if params:
            flat = {k: (json.dumps(v) if isinstance(v, list | tuple) else v) for k, v in _flat(params).items()}
            mlflow.log_params({k[:250]: str(v)[:500] for k, v in flat.items()})
        if tags:
            mlflow.set_tags({k: str(v) for k, v in tags.items()})

        def log(metrics: dict | None = None, artifact: dict | None = None, artifact_name: str = "result.json"):
            if metrics:
                mlflow.log_metrics({k: float(v) for k, v in _flat(metrics).items()
                                    if isinstance(v, int | float) and v == v})
            if artifact is not None:
                mlflow.log_dict(artifact, artifact_name)
        yield log


def log_evaluation(experiment: str, name: str, spec: dict, evaluation: dict, decision: dict | None = None,
                   tags: dict | None = None):
    """One walk-forward evaluation (eval.harness.evaluate output) and, optionally, its gate decision."""
    with run(experiment, name, params=spec, tags=tags) as log:
        m = dict(evaluation.get("summary", {}))
        m.update({f"calibration.{k}": v for k, v in evaluation.get("calibration", {}).items()})
        if decision:
            m.update({"gate.composite": decision.get("composite", float("nan")),
                      "gate.bootstrap_win_rate": decision.get("bootstrap_win_rate", float("nan")),
                      "gate.promote": float(bool(decision.get("promote")))})
            m.update({f"gate.relative.{k}": v for k, v in decision.get("relative", {}).items()})
        log(m, artifact={"spec": spec, "decision": decision}, artifact_name="evaluation.json")
