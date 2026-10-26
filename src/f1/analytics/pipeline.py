"""The analytics build as a sequence of stages (run by scripts/build_analytics.py).

Each stage reads what earlier stages put on the shared Context, writes its
tables to data/analytics/ and adds a section to report.json. Order matters:

    weekends -> circuits -> eras -> strategy -> dna -> profiles -> affinity
             -> race_pace -> projection -> prerace

Every table is validated against its data contract (f1.schemas) as it is written.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import ingest, schemas
from ..config import era_of
from . import build as B
from . import circuits as C
from . import dna as D
from . import eras as E
from . import ml, prerace
from . import strategy as S
from .telemetry import ZONE_CLASSES

OUT = B.OUT
ERA_METRICS = ["overtakes_per_lap", "pit_loss_s", "sc_rate", "race_deg", "race_deg_soft", "race_deg_medium",
               "race_deg_hard"]


@dataclass
class Context:
    tables: dict
    results: pd.DataFrame
    calendar: pd.DataFrame
    now: pd.Timestamp
    report: dict = field(default_factory=dict)
    prof: pd.DataFrame = field(default_factory=pd.DataFrame)      # circuit profiles, era-adjusted after stage_eras
    clusters: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_weekend: pd.DataFrame = field(default_factory=pd.DataFrame)
    driver_dna: pd.DataFrame = field(default_factory=pd.DataFrame)
    dna_check: dict = field(default_factory=dict)
    pace_frame: pd.DataFrame = field(default_factory=pd.DataFrame)


def write(df: pd.DataFrame, name: str, fmt: str = "%.5g", **kw):
    schemas.validate(df, name)
    df.to_csv(OUT / f"{name}.csv", index=False, float_format=fmt, **kw)


def _json(o):
    if isinstance(o, np.floating | np.integer):
        return o.item()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    raise TypeError(type(o))


# ------------------------------------------------------------------ stages

def stage_circuits(c: Context):
    c.prof = C.profiles(c.tables["weekend"])
    c.clusters = C.cluster(c.prof)
    write(c.clusters.reset_index(), "circuits")
    a = c.clusters.attrs
    c.report["circuits"] = {"k": a["k"], "silhouette": a["silhouette"], "explained": a["explained"],
                            "loadings": a["pc_loadings"]}


def stage_eras(c: Context):
    """Overtakes (Jolpica, 2019+) and OpenF1 circuit metrics, measured per regulation era and scaled."""
    res = c.results
    cur_era = era_of(int(res["season"].max()))
    cache = B.read("overtakes_by_race")
    ov = E.overtakes_table(res, cache if len(cache) else None)
    write(ov, "overtakes_by_race")
    layout = c.prof[E.LAYOUT].dropna()
    pm = E.fit_profile_model(ov, layout)
    est = {"overtakes_per_lap": E.current_estimates(ov, "per_lap", cur_era, E.predict_profile(pm, layout, cur_era))}
    write(est["overtakes_per_lap"].reset_index(), "eras_overtakes", "%.4g")
    wk = c.tables["weekend"].copy()
    wk["era"] = wk["season"].map(era_of)
    wk["race_deg"] = wk[["race_deg_soft", "race_deg_medium", "race_deg_hard"]].median(axis=1)
    wk["sc_rate"] = wk[["race_sc", "race_vsc"]].sum(axis=1, min_count=1)
    dry = wk[wk["race_rain_share"].fillna(0) == 0]
    for m, src in [("pit_loss_s", wk), ("sc_rate", wk), ("avg_speed_kmh", wk), ("race_deg", dry),
                   ("race_deg_soft", dry), ("race_deg_medium", dry), ("race_deg_hard", dry)]:
        est[m] = E.current_estimates(src.dropna(subset=[m]), m, cur_era)
    write(pd.concat([e.reset_index().assign(metric=m) for m, e in est.items()], ignore_index=True),
          "eras_metrics", "%.4g")
    rep = {"current_era": cur_era, "current_era_name": E.era_name(int(res["season"].max())),
           "factors": {m: {k: v for k, v in e.attrs["factor"].items() if k != "per_circuit"} for m, e in est.items()}}
    if pm is not None:
        n = len(E.LAYOUT)
        rep["profile_model"] = {"coef_per_sd": dict(zip(E.LAYOUT, np.round(pm["model"].coef_[:n], 3))),
                                "era_terms": dict(zip([str(e) for e in pm["eras"][1:]],
                                                      np.round(np.exp(pm["model"].coef_[n:]), 3)))}
    ev, rep["validation"] = E.validate(ov, layout, cur_era)
    write(ev, "eras_check", "%.4g")
    c.report["eras"] = rep
    # strategy and projections use current-era values (previous era x factor where not raced yet)
    prof = c.prof.copy()
    for m in ERA_METRICS:
        e = est[m]["estimate"]
        prof[m] = e.reindex(prof.index).fillna(prof[m]) if m in prof else e.reindex(prof.index)
    c.prof = prof


def stage_strategy(c: Context):
    plans = []
    for cid, row in c.prof.iterrows():
        sp = S.for_circuit(row, c.prof)
        if sp is not None:
            plans.append(sp.head(6).assign(circuit_id=cid, race_name=row.race_name).drop(columns=["compounds", "stints"]))
    if plans:
        write(pd.concat(plans, ignore_index=True), "strategy", "%.4g")
    sv, sres = S.validate(c.tables["weekend"])
    if len(sv):
        write(sv, "strategy_check")
        c.report["strategy"] = sres


def stage_dna(c: Context):
    t = c.tables
    tw = D.team_weekend(t["zones"], t["one_lap"], t["race_pace"], t["long_runs"])
    write(tw, "team_weekend")
    write(D.rolling_dna(tw), "team_dna_rolling")
    m, c.dna_check = D.validate_projection(tw, t["weekend"])
    write(m[["season", "round", "circuit_id", "constructor_id", "team_name", "team_colour", "actual", "proj_gap",
             "blend_gap", "season_avg_gap", "last_gap"]], "projection_check", "%.4g")
    c.report["dna_projection"] = c.dna_check
    seasons = sorted(t["zones"]["season"].unique())
    c.driver_dna = pd.concat([D.driver_dna(t["zones"], t["one_lap"], t["race_pace"], t.get("race_story", pd.DataFrame()),
                                           c.results, s).assign(season=s) for s in seasons], ignore_index=True)
    write(c.driver_dna, "driver_dna", "%.4g")
    cm = [x for x in (ml.car_map(tw, s).assign(season=s) for s in seasons) if len(x)]
    if cm:
        write(pd.concat(cm, ignore_index=True), "car_map", "%.4g")
    an = ml.anomalies(tw)
    write(an[an["anomaly"]].sort_values("anomaly_score", ascending=False), "anomalies", "%.4g")
    c.team_weekend = tw


def stage_profiles(c: Context):
    """Driver and car profiles (the race simulator's agents) and k-means driver archetypes."""
    from . import profiles as PR
    P = PR.build(c.tables, c.results)
    write(P, "profiles", "%.4g")
    lat = PR.archetypes(P.attrs["latest"])
    lat["code"] = lat["driver_id"].map(_codes(c.results))
    write(lat, "profiles_latest", "%.4g")


def stage_affinity(c: Context):
    """Track affinity: drivers' teammate-relative qualifying by circuit, 2014 onward (unsupervised)."""
    from . import affinity as AF
    obs = AF.observations(c.results)
    cell, wide = AF.affinity_matrix(obs)
    types = dict(zip(c.clusters.index, c.clusters["cluster_name"]))
    names = _codes(c.results)
    cell = AF.significance(obs, cell).assign(code=lambda d: d["driver_id"].map(names),
                                             circuit_type=lambda d: d["circuit_id"].map(types))
    write(cell, "affinity_cells", "%.4g")
    fd, fc, L = AF.factors(wide, 2)
    rcl, ccl = AF.coclusters(L, 4)
    write(fd.assign(cocluster=rcl, code=fd.index.map(names)).reset_index(), "affinity_drivers", "%.4g")
    write(fc.assign(cocluster=ccl, circuit_type=fc.index.map(types)).reset_index(), "affinity_circuits", "%.4g")
    tt, tg = AF.by_type(cell, types)
    write(tt.assign(track_group=tg, code=tt.index.map(names)).reset_index(), "affinity_types", "%.4g")
    c.report["affinity"] = {"holdout": AF.holdout_check(wide, 2), "temporal": AF.temporal_check(obs, types),
                            "drivers": int(wide.shape[0]), "circuits": int(wide.shape[1])}


def stage_race_pace(c: Context):
    """Gradient-boosted Sunday race-pace projection, walk-forward checked, with SHAP explanations."""
    c.pace_frame = ml.race_pace_frame(c.tables)
    if c.pace_frame["race_pace_gap_pct"].notna().sum() <= 200:
        return
    P, res = ml.race_pace_model(c.pace_frame)
    write(P, "race_pace_check", "%.4g")
    shap_tab = ml.race_pace_shap(c.pace_frame)
    if shap_tab is not None:
        write(shap_tab, "race_pace_shap", "%.4g")
        res["shap_mean_abs"] = shap_tab.groupby("feature")["shap"].apply(lambda s: float(s.abs().mean())).to_dict()
    c.report["race_pace_model"] = res


def stage_projection(c: Context):
    """Next race: circuit, strategy, team pace from DNA, pre-weekend win probabilities, race pace after quali."""
    up = c.calendar[c.calendar["race_utc"] > c.now - pd.Timedelta(hours=3)].sort_values("race_utc")
    if up.empty:
        return
    nr = up.iloc[0]
    s, r, cid = int(nr.season), int(nr["round"]), nr.circuit_id
    t, prof, cl = c.tables, c.prof, c.clusters
    proj = {"season": s, "round": r, "race_name": nr.race_name, "circuit_id": cid, "race_utc": nr.race_utc,
            "format": nr.format}
    w = t["weekend"]
    past = w[(w["circuit_id"] == cid) & ((w["season"] < s) | ((w["season"] == s) & (w["round"] < r)))]
    past = past.dropna(subset=["share_straight"]).sort_values(["season", "round"])
    if cid in prof.index:
        sp = S.for_circuit(prof.loc[cid], prof)
        if sp is not None:
            proj["strategy"] = sp.head(6).drop(columns=["compounds", "stints"]).to_dict("records")
            proj["strategy_inputs"] = sp.attrs
        proj["circuit"] = {k: prof.loc[cid, k] for k in ["overtakes_per_lap", "pit_loss_s", "race_laps", "sc_rate",
                                                         "race_track_temp", "race_deg"] if k in prof.columns}
    if cid in cl.index:
        proj["cluster"] = cl.loc[cid, "cluster_name"]
        proj["similar"] = {k: round(float(v), 2) for k, v in C.similar(cl, cid).items()}
    if len(past):
        _team_projection(c, proj, past.iloc[-1], s, r)
    wk = t["one_lap"][(t["one_lap"]["season"] == s) & (t["one_lap"]["round"] == r)]
    if len(wk):
        proj["sessions_done"] = sorted(set(wk["session"]))
        if "Q" in proj["sessions_done"] and len(c.pace_frame):
            f = c.pace_frame
            now = f[(f["season"] == s) & (f["round"] == r)]
            hist = f[(f["season"] < s) | ((f["season"] == s) & (f["round"] < r))]
            if len(now) and hist["race_pace_gap_pct"].notna().sum() > 200:
                mdl = ml.fit_race_pace(hist)
                pr = now[["driver_id", "code", "constructor_id", "quali_gap_pct"]].copy()
                pr["proj_race_pace_gap"] = mdl.predict(now[ml.PACE_INPUTS])
                pr["proj_race_pace_gap"] -= pr["proj_race_pace_gap"].min()
                proj["race_pace_projection"] = pr.sort_values("proj_race_pace_gap").to_dict("records")
    (OUT / "projections.json").write_text(json.dumps(proj, default=_json, indent=1))


def _team_projection(c: Context, proj: dict, p: pd.Series, s: int, r: int):
    from .. import registry
    from . import preweekend
    proj["profile_from"] = f"{int(p.season)} R{int(p['round'])}"
    proj["profile"] = {k: float(p[k]) for k in ["share_straight", "share_fast", "share_medium", "share_slow",
                                                "top_speed_kmh", "avg_speed_kmh", "full_throttle_pct"]}
    tw, res = c.team_weekend, c.results
    dna = D.latest_dna(tw[tw["season"] <= s])
    cur = res[res["season"] == s]
    cur = cur[cur["round"] == cur["round"].max()]["constructor_id"].unique() if len(cur) else []
    if len(cur):
        dna = dna[dna["constructor_id"].isin(cur)]
    tdf = pd.DataFrame([{"constructor_id": row.constructor_id, "team_name": row.team_name,
                         "team_colour": row.team_colour, "dna_vs_median": D.project(row, p, prefix=""),
                         "form_vs_median": float(row.get("quali_vs_median_pct", np.nan)),
                         **{f"z_{k}": float(row.get(f"z_{k}", np.nan)) for k in ZONE_CLASSES}}
                        for _, row in dna.iterrows()])
    tdf["blend_vs_median"] = 0.5 * tdf["dna_vs_median"] + 0.5 * tdf["form_vs_median"]
    for col in ["dna_vs_median", "blend_vs_median"]:
        tdf[col.replace("vs_median", "gap")] = tdf[col] - tdf[col].min()
    proj["teams"] = tdf.sort_values("blend_gap").to_dict("records")
    # pre-weekend win probabilities: Monte Carlo over qualifying + champion race model
    mae = c.dna_check.get("blend_gap", {}).get("mae_pct", 0.5)
    dd = c.driver_dna
    dds = dd[dd["season"] == s] if len(dd) and s in set(dd["season"]) else pd.DataFrame()
    split = dds.set_index("driver_id")["tm_quali_pct"] if len(dds) else pd.Series(dtype=float)
    reg = registry.ensure_initialized()
    pw = preweekend.simulate(s, r, tdf.rename(columns={"blend_gap": "gap_pct"}), split, 1.25 * mae,
                             registry.champion_spec(reg), results=res)
    proj["preweekend"] = {"model_version": reg["champion"], **pw.attrs, "drivers": pw.round(4).to_dict("records")}


def stage_prerace(c: Context):
    write(prerace.table(c.tables), "prerace_features", "%.4g")


STAGES = [stage_circuits, stage_eras, stage_strategy, stage_dna, stage_profiles, stage_affinity, stage_race_pace,
          stage_projection, stage_prerace]


def _codes(results: pd.DataFrame) -> pd.Series:
    return results.drop_duplicates("driver_id", keep="last").set_index("driver_id")["code"]


def run(verbose: bool = False) -> dict:
    tables = B.build_all()
    for name, df in tables.items():
        schemas.validate(df, name)
    c = Context(tables=tables, results=ingest.load_results(), calendar=ingest.load_calendar(),
                now=pd.Timestamp.now(tz="UTC"), report={"weekends": len(tables.get("weekend", []))})
    for stage in STAGES:
        stage(c)
        if verbose:
            print(stage.__name__, "done", flush=True)
    (OUT / "report.json").write_text(json.dumps(c.report, default=_json, indent=1))
    return c.report
