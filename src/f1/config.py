"""Paths and constants shared across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS_DIR = DATA / "results"          # one CSV per season, one row per driver-race
CALENDAR_CSV = DATA / "calendar.csv"
LAPS_DIR = DATA / "laps"                # lap-by-lap race timing for reviewed races
CACHE = ROOT / ".cache"
PREDICTIONS = ROOT / "predictions"
PRED_LOG = PREDICTIONS / "log.jsonl"
SCORES_LOG = PREDICTIONS / "scores.jsonl"
CHAMP_DIR = DATA / "championship"
MODELS = ROOT / "models"
REGISTRY = MODELS / "registry.json"
REPORTS = ROOT / "reports"
EXPERIMENTS = ROOT / "experiments"
EXP_LOG = EXPERIMENTS / "log.jsonl"
EXP_QUEUE = EXPERIMENTS / "queue.json"
PROPOSALS = ROOT / "proposals"
STATE = ROOT / "state"
LAST_RUN = STATE / "last_run.json"
CHANGELOG = ROOT / "CHANGELOG_MODELS.md"

FIRST_SEASON = 2014  # start of the hybrid era; earlier data adds little and costs API calls
SEED = 20260101

# Regulation eras (first season of each). Form from a previous era goes stale.
ERA_STARTS = (2014, 2017, 2022, 2026)

RACE_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
# Sprint points by season range; 2021 sprints paid 3-2-1.
SPRINT_POINTS_2021 = [3, 2, 1]
SPRINT_POINTS = [8, 7, 6, 5, 4, 3, 2, 1]


def era_of(season: int) -> int:
    return max(s for s in ERA_STARTS if s <= season)


def sprint_points(season: int) -> list[int]:
    return SPRINT_POINTS_2021 if season == 2021 else SPRINT_POINTS
