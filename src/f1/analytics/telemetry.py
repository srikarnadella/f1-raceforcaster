"""Where on the lap is time won and lost? Qualifying telemetry, zone by zone.

Each driver's fastest qualifying lap (OpenF1 car_data, ~3.7 Hz) is put on a
common distance grid. The pole lap defines the zones:
  straight   full throttle (>= 95%) and no brake
  corner     everything between straights, classed by the pole lap's minimum
             speed in it: slow (< 130 km/h), medium (130-210), fast (>= 210)
Flat-out kinks (e.g. Copse, 130R in some years) count as straight.

Two views per zone class:
  time   zone time from the time-vs-distance curve, so a lap's deltas add up
         to its gap to pole. Time lost on a straight includes slower exits from
         the corner before it.
  speed  apex (minimum) speed in corners and peak speed on straights, % vs the
         field median: a purer grip / drag signal that needs no alignment.
Laps are aligned on the official sector lines; telemetry is ~3.7 Hz, so
single-corner deltas carry a few hundredths of noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import openf1

GRID_M = 5.0
SLOW, FAST = 130.0, 210.0
ZONE_CLASSES = ["straight", "fast", "medium", "slow"]


def load(season: int, rnd: int) -> pd.DataFrame | None:
    """Fastest-quali-lap telemetry with that lap's official sector times attached."""
    d = openf1.meeting_dir(season, rnd)
    if not (d / "tel_Q.csv").exists():
        return None
    tel = pd.read_csv(d / "tel_Q.csv")
    laps = pd.read_csv(d / "laps.csv", usecols=["session", "driver_number", "lap_s", "s1", "s2"])
    q = laps[(laps["session"] == "Q") & laps["lap_s"].notna()]
    q = q.loc[q.groupby("driver_number")["lap_s"].idxmin(), ["driver_number", "s1", "s2"]]
    return tel.merge(q, on="driver_number", how="left")


def _extend(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, dist, speed) from the timing line (t=0) to the end of the lap (t=lap_s).
    Older files measured t from the first sample; they are treated as starting
    at the line."""
    d = d.sort_values("t")
    t, x, v = d["t"].to_numpy(float), d["dist"].to_numpy(float), d["speed"].to_numpy(float)
    lap = float(d["lap_s"].iat[0])
    t_end = lap
    x_end = x[-1] + v[-1] / 3.6 * max(t_end - t[-1], 0.0)
    if t[0] > 0:
        t, x, v = np.r_[0.0, t], np.r_[0.0, x], np.r_[v[0], v]
    keep = t <= t_end
    return np.r_[t[keep], t_end], np.r_[x[keep], x_end], np.r_[v[keep], v[-1]]


def traces(tel: pd.DataFrame) -> tuple[np.ndarray, dict[int, pd.DataFrame], int]:
    """Resample each lap onto the pole lap's distance grid.

    Each lap's integrated distance is scaled to the pole lap's length, then
    warped piecewise-linearly so the lap crosses the official sector lines
    (located on the pole lap from its S1/S2 times) at its own official S1 and
    S1+S2 times. Checked against official sectors: scaling alone is within
    ~0.03 s per sector; the anchors make sector totals exact."""
    pole_no = int(tel.loc[tel["lap_s"].idxmin(), "driver_number"])
    P = tel[tel["driver_number"] == pole_no]
    tp, xp, vp = _extend(P)
    L = xp[-1]
    lines = None
    if {"s1", "s2"} <= set(tel.columns) and P[["s1", "s2"]].notna().all(axis=None):
        s1, s2 = float(P["s1"].iat[0]), float(P["s2"].iat[0])
        lines = np.interp([s1, s1 + s2], tp, xp)
    grid = np.arange(0, L, GRID_M)
    out = {}
    for no, d in tel.groupby("driver_number"):
        # incomplete car-data streams (gaps, a few samples) cannot be placed on the lap: skip them
        if len(d) < 0.6 * len(P) or d["t"].max() - max(d["t"].min(), 0) < 0.85 * float(d["lap_s"].iat[0]):
            continue
        t, x, v = _extend(d)
        if x[-1] <= 0:
            continue
        x = x / x[-1] * L
        if lines is not None and int(no) != pole_no and d[["s1", "s2"]].notna().all(axis=None):
            s1, s2 = float(d["s1"].iat[0]), float(d["s2"].iat[0])
            own = np.interp([s1, s1 + s2], t, x)
            if 0 < own[0] < own[1] < L:
                x = np.interp(x, [0, own[0], own[1], L], [0, lines[0], lines[1], L])
        d = d.sort_values("t")
        xs = np.interp(d["t"].to_numpy(float), t, x)
        out[int(no)] = pd.DataFrame({
            "dist": grid, "t": np.interp(grid, x, t),
            "speed": np.interp(grid, xs, d["speed"].to_numpy(float)),
            "throttle": np.interp(grid, xs, d["throttle"].to_numpy(float)),
            "brake": np.interp(grid, xs, (d["brake"] > 0).to_numpy(float)) > 0.5,
            "gear": np.round(np.interp(grid, xs, d["n_gear"].to_numpy(float))),
            "lap_s": float(d["lap_s"].iat[0]),
        })
    return grid, out, pole_no


def sector_check(tel: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """Validate alignment against official timing: place the sector lines on the
    pole lap using its official S1/S2 times, then compare every driver's
    telemetry time to those points with their official sector times."""
    grid, tr, pole = traces(tel)
    q = laps[(laps["session"] == "Q") & laps["lap_s"].notna()]
    best = q.loc[q.groupby("driver_number")["lap_s"].idxmin()].set_index("driver_number")
    if pole not in best.index or best.loc[pole, ["s1", "s2"]].isna().any():
        return pd.DataFrame()
    ref = tr[pole]
    d1 = np.interp(best.loc[pole, "s1"], ref["t"], ref["dist"])
    d2 = np.interp(best.loc[pole, "s1"] + best.loc[pole, "s2"], ref["t"], ref["dist"])
    rows = []
    for no, d in tr.items():
        if no == pole or no not in best.index or best.loc[no, ["s1", "s2", "s3"]].isna().any():
            continue
        t1, t2 = np.interp([d1, d2], d["dist"], d["t"])
        b = best.loc[no]
        rows.append({"driver_number": no, "err_s1": t1 - b.s1, "err_s2": (t2 - t1) - b.s2,
                     "err_s3": (b.lap_s - t2) - b.s3})
    return pd.DataFrame(rows)


def zones(ref: pd.DataFrame, min_gap_m: float = 40.0) -> pd.DataFrame:
    """Segment the reference lap into straights and corners."""
    full = (ref["throttle"].to_numpy() >= 95) & ~ref["brake"].to_numpy()
    # close short gaps in either state (sensor flicker)
    n_gap = int(min_gap_m / GRID_M)
    full = _close(full, n_gap)
    full = ~_close(~full, n_gap)
    seg_id = np.cumsum(np.r_[0, np.diff(full.astype(int)) != 0])
    seg_id = _split_corners(seg_id, full, ref["speed"].to_numpy())
    rows = []
    for k in np.unique(seg_id):
        idx = np.where(seg_id == k)[0]
        sp = ref["speed"].to_numpy()[idx]
        kind = "straight" if full[idx[0]] else ("slow" if sp.min() < SLOW else "medium" if sp.min() < FAST else "fast")
        rows.append({"zone": int(k), "start": float(ref["dist"].iat[idx[0]]), "end": float(ref["dist"].iat[idx[-1]]) + GRID_M,
                     "i0": int(idx[0]), "i1": int(idx[-1]), "kind": kind,
                     "min_speed": float(sp.min()), "max_speed": float(sp.max())})
    z = pd.DataFrame(rows)
    corners = z["kind"] != "straight"
    z.loc[corners, "corner"] = np.arange(1, corners.sum() + 1)
    return z


def _split_corners(seg_id, full, speed, prominence=15.0):
    """A braking-to-throttle zone can hold several corners (chicanes, esses):
    split it at the speed peak between two apexes that are >= `prominence` km/h
    below that peak."""
    from scipy.signal import find_peaks
    out = np.zeros_like(seg_id)
    nxt = 0
    for k in np.unique(seg_id):
        idx = np.where(seg_id == k)[0]
        cuts = []
        if not full[idx[0]] and len(idx) > 6:
            sp = np.convolve(speed[idx], np.ones(3) / 3, mode="same")
            mins, _ = find_peaks(-sp, prominence=prominence)
            for a, b in zip(mins[:-1], mins[1:]):
                cuts.append(a + int(np.argmax(sp[a:b])))
        lab = np.searchsorted(np.array(cuts), np.arange(len(idx)), side="right")
        out[idx] = nxt + lab
        nxt += lab.max() + 1
    return out


def _close(mask: np.ndarray, n: int) -> np.ndarray:
    """Fill False runs shorter than n samples that are surrounded by True."""
    m = mask.copy()
    i = 0
    while i < len(m):
        if not m[i]:
            j = i
            while j < len(m) and not m[j]:
                j += 1
            if i > 0 and j < len(m) and j - i < n:
                m[i:j] = True
            i = j
        else:
            i += 1
    return m


def zone_table(tel: pd.DataFrame, ids: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (per-driver per-zone times, per-driver per-class summary, zones)."""
    grid, tr, pole = traces(tel)
    z = zones(tr[pole])
    rows = []
    for no, d in tr.items():
        t, sp = d["t"].to_numpy(), d["speed"].to_numpy()
        t_ext = np.r_[t, d["lap_s"].iat[0]]
        for zz in z.itertuples():
            rows.append({"driver_number": no, "zone": zz.zone, "kind": zz.kind, "corner": zz.corner,
                         "time_s": t_ext[zz.i1 + 1] - t_ext[zz.i0],
                         "min_speed": sp[zz.i0:zz.i1 + 1].min(), "max_speed": sp[zz.i0:zz.i1 + 1].max()})
    zt = pd.DataFrame(rows).merge(ids[["driver_number", "code", "driver_id", "constructor_id", "team_name", "team_colour"]],
                                  on="driver_number", how="left")
    ref = zt[zt["driver_number"] == pole].set_index("zone")
    zt["delta_to_pole_s"] = zt["time_s"] - zt["zone"].map(ref["time_s"])
    zt["apex_delta_kmh"] = zt["min_speed"] - zt["zone"].map(ref["min_speed"])
    # speed view (alignment-free): apex speed in corners, peak speed on straights, vs the field median
    key_speed = np.where(zt["kind"] == "straight", zt["max_speed"], zt["min_speed"])
    zt["key_speed"] = key_speed
    zt["speed_vs_median_pct"] = (zt["key_speed"] / zt.groupby("zone")["key_speed"].transform("median") - 1) * 100
    cls = zt.groupby(["driver_id", "code", "constructor_id", "team_name", "team_colour", "kind"]).agg(
        time_s=("time_s", "sum"), delta_to_pole_s=("delta_to_pole_s", "sum"),
        speed_vs_median_pct=("speed_vs_median_pct", "mean")).reset_index()
    med = cls.groupby("kind")["time_s"].transform("median")
    cls["vs_median_pct"] = (cls["time_s"] / med - 1) * 100
    cls["lap_s"] = cls["driver_id"].map(zt.drop_duplicates("driver_id").set_index("driver_id")["driver_number"].map(
        {k: v["lap_s"].iat[0] for k, v in tr.items()}))
    return zt, cls, z


def circuit_profile(tel: pd.DataFrame) -> dict:
    """Shape of the lap from the pole lap: where the time is spent."""
    grid, tr, pole = traces(tel)
    ref = tr[pole]
    z = zones(ref)
    t_ext = np.r_[ref["t"].to_numpy(), ref["lap_s"].iat[0]]
    tz = {k: 0.0 for k in ZONE_CLASSES}
    for zz in z.itertuples():
        tz[zz.kind] += t_ext[zz.i1 + 1] - t_ext[zz.i0]
    lap = ref["lap_s"].iat[0]
    L = grid[-1] + GRID_M
    gear = ref["gear"].to_numpy()
    return {
        "lap_km": L / 1000, "pole_lap_s": lap, "avg_speed_kmh": L / lap * 3.6,
        "top_speed_kmh": float(ref["speed"].max()), "min_speed_kmh": float(ref["speed"].min()),
        **{f"share_{k}": tz[k] / lap for k in ZONE_CLASSES},
        "full_throttle_pct": float((ref["throttle"] >= 95).mean() * 100),
        "braking_pct": float(ref["brake"].mean() * 100),
        "n_slow": int((z["kind"] == "slow").sum()), "n_medium": int((z["kind"] == "medium").sum()),
        "n_fast": int((z["kind"] == "fast").sum()),
        "gear_changes_per_km": float((np.abs(np.diff(gear)) > 0).sum() / (L / 1000)),
    }


def dominance_map(season: int, rnd: int, ids: pd.DataFrame, n_sectors: int = 30) -> pd.DataFrame | None:
    """Track outline (pole lap x/y) split into equal-distance mini-sectors, each
    tagged with the team whose fastest qualifying lap was quickest through it."""
    tel = load(season, rnd)
    p = openf1.meeting_dir(season, rnd) / "map_Q.csv"
    if tel is None or not p.exists():
        return None
    mp = pd.read_csv(p)
    grid, tr, pole = traces(tel)
    raw = tel[tel["driver_number"] == pole].sort_values("t")
    # x/y samples -> distance along the pole lap
    mp["dist"] = np.interp(mp["t"], raw["t"], raw["dist"] / raw["dist"].max() * grid[-1])
    edges = np.linspace(0, grid[-1], n_sectors + 1)
    team = ids.set_index("driver_number")
    rows = []
    for k in range(n_sectors):
        a, b = edges[k], edges[k + 1]
        best: tuple[int, float] = (-1, np.inf)
        for no, d in tr.items():
            ta, tb = np.interp([a, b], d["dist"], d["t"])
            if tb - ta < best[1]:
                best = (no, tb - ta)
        no = best[0]
        rows.append({"sector": k, "driver_number": no,
                     "code": team["code"].get(no, str(no)), "team_name": team["team_name"].get(no, "?"),
                     "team_colour": team["team_colour"].get(no, "#888888")})
    sec = pd.DataFrame(rows)
    mp["sector"] = np.clip(np.searchsorted(edges, mp["dist"], side="right") - 1, 0, n_sectors - 1)
    return mp.merge(sec, on="sector")


def speed_traces(season: int, rnd: int, numbers: list[int]) -> pd.DataFrame:
    tel = load(season, rnd)
    if tel is None:
        return pd.DataFrame()
    grid, tr, _ = traces(tel)
    ref = tr[min(tr, key=lambda k: tr[k]["lap_s"].iat[0])]
    out = []
    for no in numbers:
        if no in tr:
            d = tr[no].iloc[::2].copy()
            d["delta_s"] = d["t"] - np.interp(d["dist"], ref["dist"], ref["t"])
            out.append(d.assign(driver_number=no))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
