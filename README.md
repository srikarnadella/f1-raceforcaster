# F1 Forecaster

Forecasts F1 race winner, podium/top-10, finishing order and championship from qualifying (and, if F1TV credentials
are available, practice) data, then scores every forecast against reality. See `CLAUDE.md` for the full brief.
Not betting or financial advice.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest                          # leakage, model, gate, pipeline tests
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from f1 import ingest; ingest.update()"  # refresh data
.venv/bin/python scripts/backtest.py --out reports/backtest.json   # walk-forward vs baselines
.venv/bin/python scripts/should_run.py              # which trigger is due now
.venv/bin/python scripts/run_forecast.py 2026_19    # Trigger A (after qualifying)
.venv/bin/python scripts/run_review.py 2026_18      # Trigger B (after the race)
.venv/bin/python scripts/run_improve.py --dry-run   # one improvement cycle, nothing written
.venv/bin/streamlit run app/streamlit_app.py        # dashboard
```

| Path | What |
|---|---|
| `src/f1/` | ingestion (Jolpica), features, models, simulation, championship, laps, review, registry |
| `eval/` | **protected** walk-forward harness, metrics, promotion gate |
| `predictions/log.jsonl` | append-only pre-race forecasts; `scores.jsonl` their scores |
| `models/registry.json` | versions, champion and shadow pointers, history |
| `experiments/queue.json` | candidate changes for the improvement cycle; results in `experiments/log.jsonl` |
