"""Data contracts (pandera) for the stored and derived tables.

Every table written by ingest.update and the analytics pipeline is validated
here first, so a source change or a bug shows up as a failed build with the
offending rows, not as quietly wrong numbers on the dashboard. Tables without
a contract pass through unchanged.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

SEASON = Column(int, Check.in_range(2014, 2035))
ROUND = Column(int, Check.in_range(1, 30))
SESSIONS = ["FP1", "FP2", "FP3", "SQ", "S", "Q", "R"]
COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def _unique(*cols):
    return Check(lambda d: ~d.duplicated(list(cols)), element_wise=False,
                 error=f"duplicate rows for {', '.join(cols)}")


SCHEMAS: dict[str, DataFrameSchema] = {
    "results": DataFrameSchema({
        "season": SEASON, "round": ROUND,
        "driver_id": Column(str), "constructor_id": Column(str), "circuit_id": Column(str),
        "position": Column(int, Check.ge(1)),
        "grid": Column(int, Check.ge(0)),
        "points": Column(float, Check.in_range(0, 60)),
        "quali_pos": Column(float, Check.in_range(1, 30), nullable=True),
        "q1": Column(float, Check.in_range(50, 200), nullable=True),
    }, checks=[_unique("season", "round", "driver_id"),
               Check(lambda d: d.groupby(["season", "round"])["position"].transform(lambda s: (s == 1).sum()) == 1,
                     element_wise=False, error="every race has exactly one winner")]),
    "calendar": DataFrameSchema({
        "season": SEASON, "round": ROUND, "circuit_id": Column(str),
        "format": Column(str, Check.isin(["conventional", "sprint"])),
    }, checks=[_unique("season", "round")]),
    "one_lap": DataFrameSchema({
        "season": SEASON, "round": ROUND, "session": Column(str, Check.isin(SESSIONS)),
        "driver_id": Column(str), "best_s": Column(float, Check.in_range(50, 200)),
        "gap_pct": Column(float, Check.ge(0)),
    }, checks=[_unique("season", "round", "session", "driver_id")]),
    "race_pace": DataFrameSchema({
        "season": SEASON, "round": ROUND, "session": Column(str, Check.isin(["R", "S"])),
        "race_pace_gap_pct": Column(float, Check.ge(-1e-9), nullable=True),
    }),
    "long_runs": DataFrameSchema({
        "season": SEASON, "round": ROUND, "long_laps": Column(int, Check.ge(5)),
        "main_compound": Column(str, Check.isin(COMPOUNDS)),
    }),
    "weekend": DataFrameSchema({
        "season": SEASON, "round": ROUND,
        **{f"share_{k}": Column(float, Check.in_range(0, 1), nullable=True)
           for k in ("straight", "fast", "medium", "slow")},
        "pit_loss_s": Column(float, Check.in_range(5, 60), nullable=True),
    }, checks=[_unique("season", "round"),
               Check(lambda d: (d[[f"share_{k}" for k in ("straight", "fast", "medium", "slow")]].sum(axis=1)
                                .where(d["share_straight"].notna(), 1.0) - 1).abs() < 0.02,
                     element_wise=False, error="zone shares must add up to the whole lap")]),
    "zones": DataFrameSchema({
        "season": SEASON, "round": ROUND, "kind": Column(str, Check.isin(["straight", "fast", "medium", "slow"])),
        "time_s": Column(float, Check.gt(0)),
    }),
    "overtakes_by_race": DataFrameSchema({
        "season": SEASON, "round": ROUND, "overtakes": Column(int, Check.ge(0)),
        "per_lap": Column(float, Check.in_range(0, 5)),
    }, checks=[_unique("season", "round")]),
    "profiles": DataFrameSchema({
        "season": SEASON, "round": ROUND, "driver_id": Column(str),
        "car_dnf": Column(float, Check.in_range(0, 1)), "races_profiled": Column(int, Check.ge(0)),
    }, checks=[_unique("season", "round", "driver_id")]),
    "team_weekend": DataFrameSchema({
        "season": SEASON, "round": ROUND, "constructor_id": Column(str),
        "quali_gap_pct": Column(float, Check.ge(0), nullable=True),
    }, checks=[_unique("season", "round", "constructor_id")]),
    "affinity_cells": DataFrameSchema({
        "driver_id": Column(str), "circuit_id": Column(str), "count": Column(int, Check.ge(1)),
    }, checks=[_unique("driver_id", "circuit_id")]),
}


def validate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate `df` against the contract called `name` (no-op if there is none or df is empty)."""
    schema = SCHEMAS.get(name)
    if schema is None or df is None or df.empty:
        return df
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        failures = e.failure_cases[["column", "check", "failure_case"]].head(10).to_string(index=False)
        raise ValueError(f"data contract '{name}' failed:\n{failures}") from None
