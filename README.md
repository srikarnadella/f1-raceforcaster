# F1 Forecaster

[![ci](https://github.com/srikarnadella/f1-raceforcaster/actions/workflows/ci.yml/badge.svg)](https://github.com/srikarnadella/f1-raceforcaster/actions/workflows/ci.yml)

Forecasts F1 race winner, podium/top-10, finishing order and championship from practice, qualifying and sprint
data, scores every forecast against reality, and analyses race pace, tyre wear and where each car and driver gains
or loses time (2023 onward, via OpenF1). See `CLAUDE.md` for the full brief.
Not betting or financial advice.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt -r requirements-dev.txt
uv pip install --python .venv/bin/python -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu  # optional
pre-commit install                                  # ruff, mypy, unit tests on every commit
.venv/bin/python -m pytest                          # leakage, model, gate, pipeline tests
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from f1 import ingest; ingest.update()"  # refresh data
.venv/bin/python scripts/backtest.py --out reports/backtest.json   # walk-forward vs baselines
.venv/bin/python scripts/should_run.py              # which trigger is due now
.venv/bin/python scripts/run_forecast.py 2026_19    # Trigger A (after qualifying)
.venv/bin/python scripts/run_review.py 2026_18      # Trigger B (after the race)
.venv/bin/python scripts/run_improve.py --dry-run   # one improvement cycle, nothing written
.venv/bin/python scripts/fetch_openf1.py            # OpenF1 weekends 2023+ (resumable; ~1 min per weekend)
.venv/bin/python scripts/build_analytics.py         # analytics tables, DNA, circuit clusters, projections
.venv/bin/python scripts/model_lab.py [--gate]      # neural / practice-informed models vs champion, walk-forward
.venv/bin/streamlit run app/streamlit_app.py        # dashboard
```

| Path | What |
|---|---|
| `src/f1/` | ingestion (Jolpica, OpenF1), features, models, simulation, championship, laps, review, registry |
| `src/f1/analytics/` | lap classification and pace fits, telemetry zones, team/driver DNA, circuit clustering, ML (neural ranker, boosting, mixtures, isolation forest) |
| `eval/` | **protected** walk-forward harness, metrics, promotion gate |
| `predictions/log.jsonl` | append-only pre-race forecasts; `scores.jsonl` their scores |
| `models/registry.json` | versions, champion and shadow pointers, history |
| `experiments/queue.json` | candidate changes for the improvement cycle; results in `experiments/log.jsonl` |

## Data

Everything the models use is committed under `data/`: race results and qualifying (`data/results/`), calendar,
per-race lap timings (`data/laps/`), OpenF1 weekends 2023+ (`data/openf1/<season>_<round>/`: every session's
laps with compounds, weather, race control, qualifying telemetry), derived analytics (`data/analytics/`), and
championship history. Raw API responses are cached in `.cache/`
(gitignored, disposable). Current status, work in progress and the planned roadmap are kept in `CLAUDE.md` section 2.

## Analytics (dashboard → Analytics / Models)

- **Weekend lab**: one-lap pace and ideal laps per session; long-run pace corrected for compound and tyre age;
  tyre degradation by team and compound; unsupervised run-programme clustering; telemetry speed traces;
  time lost per zone type; track-dominance map.
- **Race story**: lap chart, gap to leader, tyre strategy, fuel/tyre-corrected clean-air race pace, race control.
- **Team DNA**: each car's strength in straights and slow/medium/fast corners (time and speed views),
  Saturday vs Sunday, car-character map (PCA).
- **Driver DNA**: teammate-relative quali, race pace, zone deltas, opening-lap gains.
- **Circuit types**: k-means clustering of circuit profiles; similar circuits.
- **Projections**: next race team pace from DNA × circuit profile; Sunday race pace from Saturday (boosting);
  logged win probabilities from the champion and challengers.
- **ML lab**: walk-forward comparison of every learned component against baselines.

## Engineering

| | |
|---|---|
| Packaging | `pyproject.toml`; `f1` and `eval` install as packages (`pip install -e .`, included in `requirements.txt`) |
| Quality gates | ruff (lint), mypy (types), pre-commit hooks; GitHub Actions `ci.yml` runs lint, type check and tests with coverage on every push and PR |
| Tests | `pytest` (leakage, models, gate, data contracts, analytics methods); `pytest -m "not data"` runs the self-contained unit tests in ~1 s |
| Data contracts | pandera schemas (`src/f1/schemas.py`) validate every stored and derived table as it is written |
| Experiment tracking | MLflow (`src/f1/tracking.py`): every gate evaluation, lab run and calibration trial; `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| Hyperparameter search | Optuna TPE for the race simulator (`scripts/race_sim_lab.py`) |
| Explainability | SHAP for the race-pace model and the 2026 confidence model |
| Deep learning | PyTorch listwise ranker (`src/f1/analytics/torch_ranker.py`), optional backend of the neural challenger |
| Pipeline | `src/f1/analytics/pipeline.py`: staged analytics build (circuits → eras → strategy → DNA → profiles → affinity → race pace → projection) |

## Deployment

- **Dashboard**: Streamlit Community Cloud, entrypoint `app/streamlit_app.py`, Python 3.12, dependencies from
  `app/requirements.txt`. It redeploys on every push to `main` (including the bot's forecast and review commits).
- **CI** (`ci.yml`): lint, type check, tests with coverage (including a render of every dashboard page) on push and PR.
- **CD check** (`deploy-check.yml`): after each push and daily, confirms the live app answers (needs the repository
  variable `STREAMLIT_APP_URL`).
- **Scheduled pipeline** (`race-weekend.yml`, `backfill-laps.yml`): forecasts after qualifying, reviews after the race,
  improvement cycle, data backfill. **Dependabot** opens weekly dependency updates.
