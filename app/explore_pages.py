"""Explore: drivers, constructors and tracks, for any season, race or circuit type.

Results and qualifying come from the Jolpica store (2014 onward); pace, zones,
tyres and overtakes from the OpenF1 analytics tables (2023 onward).
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ui
from analytics_pages import ZONE_LABEL, ZONES, read, team_scale
from f1 import ingest
from f1.analytics import telemetry as T
from f1.analytics import weekend as W

ROOT = Path(__file__).resolve().parents[1]
ALL_TYPES = "All circuits"


@st.cache_data(ttl=600)
def results():
    r = ingest.load_results()
    r["finish"] = np.where(r["classified"].astype(bool), r["position"], np.nan)
    return r


@st.cache_data(ttl=600)
def colours():
    """constructor_id -> team colour and display name, latest OpenF1 season first."""
    ol = read("one_lap")
    if ol.empty:
        return {}, {}
    o = ol.sort_values(["season", "round"]).drop_duplicates("constructor_id", keep="last")
    return dict(zip(o["constructor_id"], o["team_colour"])), dict(zip(o["constructor_id"], o["team_name"]))


def circuit_types():
    c = read("circuits")
    return dict(zip(c["circuit_id"], c["cluster_name"])) if len(c) else {}


def scope_controls(key, seasons, extra=None):
    """Season and circuit-type scope, shared by the explore screens."""
    cols = st.columns([1, 1.4] + ([1.6] if extra else []))
    season = cols[0].selectbox("Season", seasons, key=f"{key}_season")
    types = circuit_types()
    opts = [ALL_TYPES] + sorted(set(types.values()))
    ctype = cols[1].selectbox("Circuit type", opts, key=f"{key}_type")
    picked = extra(cols[2]) if extra else None
    keep = None if ctype == ALL_TYPES else {c for c, t in types.items() if t == ctype}
    return season, ctype, keep, picked


def _filter(df, season, keep):
    d = df[df["season"] == season] if "season" in df else df
    if keep is not None and "circuit_id" in d:
        d = d[d["circuit_id"].isin(keep)]
    return d


# ------------------------------------------------------------------ driver

def driver_page():
    r = results()
    seasons = sorted(r["season"].unique(), reverse=True)
    season = st.session_state.get("dr_season", seasons[0])
    pool = r[r["season"] == season]
    pts = pool.groupby(["driver_id", "code"])["points"].sum().sort_values(ascending=False).reset_index()
    names = {d: f"{c} · {d.replace('_', ' ').title()}" for d, c in zip(pts["driver_id"], pts["code"])}
    head = st.container()

    def pick(col):
        return col.selectbox("Driver", list(pts["driver_id"]), format_func=lambda d: names.get(d, d), key="dr_driver")
    season, ctype, keep, drv = scope_controls("dr", seasons, pick)
    d = _filter(r[r["driver_id"] == drv], season, keep).sort_values("round")
    if d.empty:
        st.info("No races for this driver in this scope.")
        return
    team = d["constructor_id"].mode().iat[0]
    col, tname = colours()
    mates = _filter(r[(r["constructor_id"] == team) & (r["driver_id"] != drv)], season, keep)
    h2h = d[["round", "quali_pos", "finish", "points"]].merge(
        mates[["round", "quali_pos", "finish", "points"]], on="round", suffixes=("", "_tm"))
    q_w = int((h2h["quali_pos"] < h2h["quali_pos_tm"]).sum())
    r_w = int(((h2h["finish"] < h2h["finish_tm"]) | (h2h["finish"].notna() & h2h["finish_tm"].isna())).sum())
    n = len(h2h)
    rp = read("race_pace")
    tm_race = np.nan
    if len(rp):
        rp = _filter(rp[rp["session"] == "R"], season, keep)
        mine = rp[rp["driver_id"] == drv][["round", "constructor_id", "race_pace_gap_pct"]]
        other = rp[(rp["constructor_id"] == team) & (rp["driver_id"] != drv)][["round", "race_pace_gap_pct"]]
        j = mine.merge(other, on="round", suffixes=("", "_tm")).dropna()
        tm_race = float((j["race_pace_gap_pct"] - j["race_pace_gap_pct_tm"]).mean()) if len(j) else np.nan
    finding = (f"Out-qualified the teammate in {q_w} of {n} races and finished ahead in {r_w} of {n}"
               + (f"; race pace {abs(tm_race):.2f}% {'faster' if tm_race < 0 else 'slower'} than the teammate on average."
                  if pd.notna(tm_race) else "."))
    with head:
        ui.title(f"{drv.replace('_', ' ').title()} · {tname.get(team, team)}", finding,
                 scope=f"Driver · {season} · {ctype} · {len(d)} races")
    wins = int((d["position"] == 1).sum())
    pods = int((d["position"] <= 3).sum())
    ui.kpis([("Points", f"{d['points'].sum():.0f}", None, f"{d['points'].mean():.1f} per race"),
             ("Wins · podiums", f"{wins} · {pods}"),
             ("Average qualifying", ui.dash(d["quali_pos"].mean(), "P{:.1f}")),
             ("Average finish", ui.dash(d["finish"].mean(), "P{:.1f}"), None,
              f"{int((~d['classified'].astype(bool)).sum())} not classified"),
             ("Qualifying vs teammate", f"{q_w}–{n - q_w}"),
             ("Race result vs teammate", f"{r_w}–{n - r_w}")])

    _profile_card(drv)
    a, b = st.columns([3, 2], gap="medium")
    with a:
        st.subheader("Grid to flag, race by race")
        g = d.assign(race=d["round"].astype(str) + " " + d["circuit_id"])
        g["gain"] = g["grid"].where(g["grid"] > 0) - g["finish"]
        base = alt.Chart(g).encode(x=alt.X("round:O", title="round"))
        rule = base.mark_rule(strokeWidth=2, color=ui.QUIET).encode(y=alt.Y("grid:Q", scale=alt.Scale(reverse=True, domain=[1, 22]),
                                                                             title="position"), y2="finish:Q")
        start = base.mark_point(shape="circle", filled=False, size=50, color=ui.MUTED).encode(y="grid:Q")
        end = base.mark_point(shape="circle", filled=True, size=70, color=col.get(team, ui.ACCENT)).encode(
            y="finish:Q", tooltip=["race_name", "grid", "finish", "points", "status"])
        st.altair_chart((rule + start + end).properties(height=260), width="stretch")
        ui.note("Hollow = grid slot, filled = finishing position; a line is places gained or lost. "
                "No filled dot = not classified.")
    with b:
        st.subheader("Qualifying gap to teammate")
        q = read("one_lap")
        if len(q):
            q = _filter(q[q["session"] == "Q"], season, keep)
            me = q[q["driver_id"] == drv][["round", "best_s"]]
            tm = q[(q["constructor_id"] == team) & (q["driver_id"] != drv)][["round", "best_s"]]
            j = me.merge(tm, on="round", suffixes=("", "_tm"))
            j["gap_pct"] = (j["best_s"] / j["best_s_tm"] - 1) * 100
            j["ahead"] = np.where(j["gap_pct"] < 0, "faster", "slower")
            if len(j):
                st.altair_chart(alt.Chart(j).mark_bar().encode(
                    x=alt.X("round:O"), y=alt.Y("gap_pct:Q", title="% vs teammate (negative = faster)"),
                    color=alt.Color("ahead:N", scale=alt.Scale(domain=["faster", "slower"], range=[ui.ACCENT, ui.QUIET]),
                                    legend=None),
                    tooltip=["round", alt.Tooltip("gap_pct:Q", format="+.3f")]).properties(height=260), width="stretch")
                ui.note(f"Median {j['gap_pct'].median():+.3f}% over {len(j)} qualifying sessions (fastest lap of the session).")
            else:
                st.info("No shared OpenF1 qualifying sessions in this scope (2023 onward).")

    z = read("zones")
    if len(z):
        z = _filter(z, season, keep)
        mine = z[z["driver_id"] == drv]
        other = z[(z["constructor_id"] == team) & (z["driver_id"] != drv)]
        j = mine.merge(other[["round", "kind", "time_s"]], on=["round", "kind"], suffixes=("", "_tm"))
        if len(j):
            j["pct"] = (j["time_s"] / j["time_s_tm"] - 1) * 100
            s = j.groupby("kind")["pct"].median().reindex(ZONES).reset_index()
            s["zone"] = s["kind"].map(ZONE_LABEL)
            c1, c2 = st.columns([2, 3], gap="medium")
            with c1:
                st.subheader("Where the lap is won against the teammate")
                st.altair_chart(alt.Chart(s).mark_bar().encode(
                    y=alt.Y("zone:N", sort=[ZONE_LABEL[k] for k in ZONES], title=None),
                    x=alt.X("pct:Q", title="% time vs teammate (negative = faster)"),
                    color=alt.condition("datum.pct < 0", alt.value(ui.ACCENT), alt.value(ui.QUIET)),
                    tooltip=["zone", alt.Tooltip("pct:Q", format="+.3f")]).properties(height=180), width="stretch")
                ui.note("Median over qualifying laps, same car, so the difference is the driver (and set-up).")
            with c2:
                _driver_table(d)
            return
    _driver_table(d)


TRAIT_LABEL = {"quali_skill": "One-lap pace", "race_skill": "Race pace", "tyre_skill": "Tyre management",
               "consistency": "Consistency", "start_skill": "Starts", "attack": "Attacking", "defence": "Defending"}


def _profile_card(drv):
    lat = read("profiles_latest")
    if lat.empty or drv not in set(lat["driver_id"]):
        return
    row = lat[lat["driver_id"] == drv].iloc[0]
    st.subheader("Driver profile")
    z = pd.DataFrame({"trait": [TRAIT_LABEL[k] for k in TRAIT_LABEL],
                      "z": [row.get(f"z_{k}", np.nan) for k in TRAIT_LABEL]}).dropna()
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        if len(z):
            st.altair_chart(alt.Chart(z).mark_bar().encode(
                y=alt.Y("trait:N", sort=list(TRAIT_LABEL.values()), title=None),
                x=alt.X("z:Q", title="vs the current grid (standard deviations, right = better)", scale=alt.Scale(domain=[-2.5, 2.5])),
                color=alt.condition("datum.z > 0", alt.value(ui.ACCENT), alt.value(ui.QUIET)),
                tooltip=["trait", alt.Tooltip("z:Q", format="+.2f")]).properties(height=210), width="stretch")
        else:
            st.info("Fewer than 8 profiled races: no archetype yet.")
    with c2:
        st.markdown(f"**Archetype:** {row.get('archetype', '–')}")
        st.markdown(f"Profiled over **{int(row['races_profiled'])}** races (2023 onward). Every trait is measured against "
                    "the teammate in the same car, then compared with the grid. These profiles are the 'agents' the "
                    "race simulator races against each other.")
        ui.note(f"Car: race pace {row['car_race']:.2f}% off the best car, one-lap {row['car_quali']:.2f}%, "
                f"retirement rate {row['car_dnf']:.0%}.")


def _driver_table(d):
    st.subheader("Races")
    t = d[["round", "race_name", "quali_pos", "grid", "position_text", "points", "status"]].rename(columns={
        "round": "Rd", "race_name": "Race", "quali_pos": "Quali", "grid": "Grid", "position_text": "Result",
        "points": "Pts", "status": "Status"})
    st.dataframe(t, hide_index=True, width="stretch", height=min(420, 36 + 35 * len(t)))


# ------------------------------------------------------------------ constructor

def constructor_page():
    r = results()
    seasons = sorted(r["season"].unique(), reverse=True)
    season = st.session_state.get("co_season", seasons[0])
    pts = r[r["season"] == season].groupby("constructor_id")["points"].sum().sort_values(ascending=False)
    col, tname = colours()
    head = st.container()

    def pick(c):
        return c.selectbox("Constructor", list(pts.index), format_func=lambda t: tname.get(t, t), key="co_team")
    season, ctype, keep, team = scope_controls("co", seasons, pick)
    d = _filter(r[r["constructor_id"] == team], season, keep)
    if d.empty:
        st.info("No races in this scope.")
        return
    tw = read("team_weekend")
    tws = _filter(tw, season, None) if len(tw) else tw
    mine = tws[tws["constructor_id"] == team] if len(tws) else tws
    types = circuit_types()
    finding = None
    if len(mine) and types:
        m = mine.assign(ctype=mine["circuit_id"].map(types))
        by = m.groupby("ctype")["quali_gap_pct"].mean().dropna()
        if len(by) >= 2:
            finding = (f"Closest to the front at {by.idxmin()} circuits ({by.min():.2f}% off the best team in qualifying), "
                       f"furthest at {by.idxmax()} ({by.max():.2f}%).")
    races = d.groupby("round").agg(points=("points", "sum"), best=("position", "min"))
    with head:
        ui.title(tname.get(team, team), finding, scope=f"Constructor · {season} · {ctype} · {len(races)} races")
    dm = mine if keep is None or not len(mine) else mine[mine["circuit_id"].isin(keep)]
    ui.kpis([("Points", f"{races['points'].sum():.0f}", None, f"{races['points'].mean():.1f} per race"),
             ("Wins · podiums", f"{int((d['position'] == 1).sum())} · {int((d['position'] <= 3).sum())}"),
             ("Qualifying gap to best team", ui.dash(dm["quali_gap_pct"].mean() if len(dm) else np.nan, unit="%")),
             ("Race pace gap to best team", ui.dash(dm["race_pace_gap_pct"].mean() if len(dm) and "race_pace_gap_pct" in dm else np.nan,
                                                    unit="%")),
             ("Double points finishes", f"{int((d[d['points'] > 0].groupby('round').size() == 2).sum())}")])
    if len(dm):
        a, b = st.columns([3, 2], gap="medium")
        with a:
            st.subheader("Pace through the season")
            metric = st.segmented_control("Measure", ["Qualifying", "Race pace"], default="Qualifying", key="co_m",
                                          label_visibility="collapsed")
            ycol = "quali_gap_pct" if metric != "Race pace" else "race_pace_gap_pct"
            st.altair_chart(alt.Chart(dm.dropna(subset=[ycol])).mark_line(point=True, color=col.get(team, ui.ACCENT)).encode(
                x=alt.X("round:Q", title="round", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y(f"{ycol}:Q", title="% gap to best team", scale=alt.Scale(zero=True)),
                tooltip=["round", "circuit_id", alt.Tooltip(f"{ycol}:Q", format=".2f")]).properties(height=240), width="stretch")
        with b:
            st.subheader("By circuit type")
            if types:
                m = mine.assign(ctype=mine["circuit_id"].map(types)).groupby("ctype")["quali_gap_pct"].agg(["mean", "count"]).reset_index()
                st.altair_chart(alt.Chart(m).mark_bar(color=col.get(team, ui.ACCENT)).encode(
                    y=alt.Y("ctype:N", sort="x", title=None), x=alt.X("mean:Q", title="avg quali gap to best team (%)"),
                    tooltip=["ctype", alt.Tooltip("mean:Q", format=".2f"), "count"]).properties(height=240), width="stretch")
                ui.note("All circuits in the season, whatever the circuit-type filter.")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.subheader("Car strengths vs field")
            zc = [f"z_{k}" for k in ZONES]
            s = dm[zc].mean().rename(lambda c: ZONE_LABEL[c[2:]]).reset_index()
            s.columns = ["zone", "pct"]
            st.altair_chart(alt.Chart(s).mark_bar().encode(
                y=alt.Y("zone:N", sort=[ZONE_LABEL[k] for k in ZONES], title=None),
                x=alt.X("pct:Q", title="% time vs field median (negative = faster)"),
                color=alt.condition("datum.pct < 0", alt.value(ui.ACCENT), alt.value(ui.QUIET)),
                tooltip=["zone", alt.Tooltip("pct:Q", format="+.3f")]).properties(height=180), width="stretch")
        with c2:
            st.subheader("Tyre wear in races vs field")
            dg = read("degradation")
            if len(dg):
                dg = _filter(dg[dg["session"] == "R"], season, keep)
                field = dg.groupby("compound")["deg_s_per_lap"].median()
                me = dg[dg["constructor_id"] == team].groupby("compound")["deg_s_per_lap"].median()
                t = pd.DataFrame({"team": me, "field": field}).dropna().reindex([c for c in W.COMPOUND_ORDER if c in me.index])
                t["vs_field"] = t["team"] - t["field"]
                st.dataframe(t.style.format("{:+.3f}"), width="stretch")
                ui.note("Seconds per lap of tyre age, fuel-corrected, median over races. Negative vs field = gentler on tyres.")
    st.subheader("Drivers")
    by = d.groupby(["driver_id", "code"]).agg(races=("round", "nunique"), points=("points", "sum"),
                                              avg_quali=("quali_pos", "mean"), avg_finish=("finish", "mean"),
                                              best=("position", "min")).reset_index().sort_values("points", ascending=False)
    by = by.drop(columns="driver_id").rename(columns={"code": "Driver", "races": "Races", "points": "Points",
                                                      "avg_quali": "Avg quali", "avg_finish": "Avg finish", "best": "Best"})
    st.dataframe(by.style.format({"Points": "{:.0f}", "Avg quali": "{:.1f}", "Avg finish": "{:.1f}", "Best": "{:.0f}"},
                                 na_rep="–"), hide_index=True, width="stretch")


# ------------------------------------------------------------------ track

def track_page():
    r = results()
    cal = ingest.load_calendar()
    prof = read("circuits")
    names = cal.sort_values("season").drop_duplicates("circuit_id", keep="last").set_index("circuit_id")["race_name"]
    recent = r[r["season"] >= 2023]["circuit_id"].value_counts().index
    opts = list(recent) + [c for c in names.index if c not in set(recent)]
    head = st.container()
    cid = st.selectbox("Track", opts, format_func=lambda c: names.get(c, c), key="tr_c")
    h = r[r["circuit_id"] == cid]
    _, tname = colours()
    p = prof.set_index("circuit_id").loc[cid] if len(prof) and cid in set(prof["circuit_id"]) else None
    wins = h[h["position"] == 1]
    from_pole = int((wins["grid"] == 1).sum())
    finding = (f"The winner started from pole in {from_pole} of {len(wins)} races here since {int(h['season'].min())}"
               + (f"; {p['overtakes_per_lap']:.2f} green-flag passes per lap, which makes it a "
                  f"{p['cluster_name']} circuit." if p is not None and pd.notna(p.get("overtakes_per_lap")) else "."))
    with head:
        ui.title(names.get(cid, cid), finding, scope=f"Track · {h['season'].nunique()} races since {int(h['season'].min())}")
    if p is not None:
        ui.kpis([("Pole-lap average speed", ui.dash(p.get("avg_speed_kmh"), "{:.0f}", " km/h")),
                 ("Full throttle", ui.dash(p.get("full_throttle_pct"), "{:.0f}", "%")),
                 ("Overtakes per lap", ui.dash(p.get("overtakes_per_lap"))),
                 ("Pit-stop loss", ui.dash(p.get("pit_loss_s"), "{:.1f}", " s")),
                 ("SC / VSC per race", ui.dash(p.get("sc_rate"), "{:.1f}")),
                 ("Tyre wear", ui.dash(p.get("race_deg"), "{:.3f}", " s/lap"))])
    a, b = st.columns([3, 2], gap="medium")
    with a:
        st.subheader("Winners and pole sitters")
        pole = h[h["quali_pos"] == 1][["season", "code"]].rename(columns={"code": "pole"})
        w = wins[["season", "code", "constructor_id", "grid"]].rename(columns={"code": "winner", "grid": "from_grid"})
        t = w.merge(pole, on="season", how="left").sort_values("season", ascending=False)
        t["constructor_id"] = t["constructor_id"].map(lambda c: tname.get(c, c.replace("_", " ").title()))
        t = t.rename(columns={"season": "Season", "winner": "Winner", "constructor_id": "Team", "from_grid": "Started",
                              "pole": "Pole"})[["Season", "Winner", "Team", "Started", "Pole"]]
        st.dataframe(t, hide_index=True, width="stretch",
                     height=min(400, 36 + 35 * len(t)))
    with b:
        st.subheader("Where the winner started")
        g = wins.assign(grid=wins["grid"].clip(lower=1)).groupby("grid").size().rename("wins").reset_index()
        st.altair_chart(alt.Chart(g).mark_bar().encode(
            x=alt.X("grid:O", title="grid position"), y=alt.Y("wins:Q", title="wins", axis=alt.Axis(tickMinStep=1)),
            tooltip=["grid", "wins"]).properties(height=240), width="stretch")
    tw = read("team_weekend")
    if len(tw):
        t = tw[tw["circuit_id"] == cid]
        if len(t):
            c1, c2 = st.columns([2, 3], gap="medium")
            with c1:
                st.subheader("Qualifying order here, by year")
                t = t.assign(rank=t.groupby("season")["quali_gap_pct"].rank())
                st.altair_chart(alt.Chart(t).mark_line(point=True).encode(
                    x=alt.X("season:O"), y=alt.Y("rank:Q", scale=alt.Scale(reverse=True), title="team rank"),
                    color=alt.Color("team_name:N", scale=team_scale(t), legend=None),
                    tooltip=["season", "team_name", alt.Tooltip("quali_gap_pct:Q", format=".2f")]).properties(height=260),
                    width="stretch")
            with c2:
                last = int(t["season"].max())
                rnd = int(t[t["season"] == last]["round"].iat[0])
                st.subheader(f"Fastest team by mini-sector, {last}")
                try:
                    w = W.load(last, rnd)
                    m = T.dominance_map(last, rnd, w["drivers"])
                except Exception:
                    m = None
                if m is not None:
                    st.altair_chart(alt.Chart(m).mark_circle(size=40).encode(
                        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
                        color=alt.Color("team_name:N", scale=team_scale(m), title=None),
                        tooltip=["sector", "code", "team_name"]).properties(height=300), width="stretch")
    sp = read("strategy")
    if len(sp) and cid in set(sp["circuit_id"]):
        st.subheader("Projected strategy")
        st.dataframe(sp[sp["circuit_id"] == cid][["stops", "plan", "delta_s", "pace_only_delta_s"]].rename(columns={
            "stops": "Stops", "plan": "Plan (tyre + laps)", "delta_s": "vs best, with track position (s)", "pace_only_delta_s": "vs best, pure pace (s)"}).style.format(
            {"vs best, with track position (s)": "{:.1f}", "vs best, pure pace (s)": "{:.1f}"}), hide_index=True, width="stretch")


# ------------------------------------------------------------------ regulation eras

def eras_page():
    from analytics_pages import read_json
    from f1.analytics import eras as E
    rep = read_json("report").get("eras", {})
    ov = read("overtakes_by_race")
    est = read("eras_overtakes")
    if not rep or ov.empty:
        st.info("Era analysis not built yet (`scripts/build_analytics.py`).")
        return
    f = rep["factors"]["overtakes_per_lap"]
    old_name = E.ERA_NAMES.get(f.get("old_era"), f.get("old_era"))
    finding = None
    if f.get("factor") == f.get("factor") and f.get("factor") is not None:
        raw = f.get("raw_factor", f["factor"])
        finding = (f"At the same tracks, the {rep['current_era_name']} produce {raw:.2f}× the green-flag "
                   f"passes per lap of the {old_name} cars (90% interval {f['lo']:.2f}–{f['hi']:.2f}, "
                   f"{f['circuits']} tracks raced under both). Predictions use a cautious {f['factor']:.2f}×, "
                   "shrunk toward no change while few tracks have been raced under both.")
    ui.title("Regulation eras", finding, scope="Explore · each rule set measured separately, then scaled")
    labels = {"overtakes_per_lap": "Passes per lap", "pit_loss_s": "Pit-stop loss", "sc_rate": "SC / VSC per race",
              "race_deg": "Tyre wear", "avg_speed_kmh": "Lap speed"}
    cells = []
    for k, lab in labels.items():
        v = rep["factors"].get(k, {})
        ok = v.get("factor") == v.get("factor") and v.get("factor") is not None
        cells.append((lab, f"{v['factor']:.2f}×" if ok else "–", None,
                      f"{v['circuits']} tracks, {v['lo']:.2f}–{v['hi']:.2f}" if ok else "needs races under both rule sets"))
    ui.kpis(cells)
    ui.note("Each figure is new rules ÷ previous rules at the same circuits (geometric mean, bootstrap 90% interval). "
            "Telemetry-based metrics need OpenF1 data for both eras.")

    ov["era_name"] = ov["season"].map(E.era_name)
    a, b = st.columns([3, 2], gap="medium")
    with a:
        st.subheader("Passes per lap, same tracks, old rules vs new")
        both = est.dropna(subset=["current_era_mean", "previous_era_mean"]).copy()
        if len(both):
            both["track"] = both["circuit_id"].str.replace("_", " ")
            order = list(both.sort_values("current_era_mean", ascending=False)["track"])
            base = alt.Chart(both).encode(y=alt.Y("track:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False)))
            rule = base.mark_rule(color=ui.QUIET, strokeWidth=2).encode(x="previous_era_mean:Q", x2="current_era_mean:Q")
            old_p = base.mark_point(filled=False, size=60, color=ui.MUTED).encode(
                x=alt.X("previous_era_mean:Q", title="green-flag passes per lap"),
                tooltip=["track", alt.Tooltip("previous_era_mean:Q", format=".2f")])
            new_p = base.mark_point(filled=True, size=70, color=ui.ACCENT).encode(
                x="current_era_mean:Q", tooltip=["track", alt.Tooltip("current_era_mean:Q", format=".2f")])
            st.altair_chart((rule + old_p + new_p).properties(height=max(220, 22 * len(both))), width="stretch")
            ui.note(f"Hollow = {old_name}, filled = {rep['current_era_name']}.")
    with b:
        st.subheader("Passes per lap by season")
        s = ov.groupby(["season", "era_name"])["per_lap"].mean().reset_index()
        st.altair_chart(alt.Chart(s).mark_bar().encode(
            x=alt.X("season:O", title=None), y=alt.Y("per_lap:Q", title="mean passes per lap"),
            color=alt.Color("era_name:N", title=None), tooltip=["season", "era_name", alt.Tooltip("per_lap:Q", format=".2f")])
            .properties(height=260), width="stretch")
        pm = rep.get("profile_model")
        if pm:
            st.markdown("**What makes a track good for passing** (Poisson model, effect per standard deviation)")
            co = pd.DataFrame({"feature": list(pm["coef_per_sd"]), "effect": list(pm["coef_per_sd"].values())})
            co["feature"] = co["feature"].map({"share_straight": "time on straights", "share_slow": "time in slow corners",
                                               "share_medium": "time in medium corners", "share_fast": "time in fast corners",
                                               "top_speed_kmh": "top speed", "braking_pct": "time braking"})
            co["effect"] = (np.exp(co["effect"]) - 1) * 100
            st.altair_chart(alt.Chart(co).mark_bar().encode(
                y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                x=alt.X("effect:Q", title="% more passes per +1 SD"),
                color=alt.condition("datum.effect > 0", alt.value(ui.ACCENT), alt.value(ui.QUIET)),
                tooltip=["feature", alt.Tooltip("effect:Q", format="+.1f")]).properties(height=200), width="stretch")

    st.subheader(f"Every track under the {rep['current_era_name']}")
    cal = ingest.load_calendar()
    last = int(cal["season"].max())
    done = set(results().query("season == @last")["round"])
    upcoming = cal[(cal["season"] == last) & ~cal["round"].isin(done)]
    t = est.copy()
    t["next"] = t["circuit_id"].map(dict(zip(upcoming["circuit_id"], upcoming["round"])))
    t["track"] = t["circuit_id"].map(dict(zip(cal["circuit_id"], cal["race_name"]))).fillna(t["circuit_id"])
    t = t.sort_values(["next", "estimate"], ascending=[True, False], na_position="last")
    for c in ["next", "estimate", "current_era_mean", "previous_era_mean", "scaled_previous", "profile_model"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    show = t[["track", "next", "estimate", "current_era_mean", "current_era_races", "previous_era_mean",
              "scaled_previous", "profile_model", "method"]].rename(columns={
        "track": "Track", "next": "Upcoming round", "estimate": "Estimate (passes/lap)",
        "current_era_mean": "Seen under these rules", "current_era_races": "Races", "previous_era_mean": "Previous rules",
        "scaled_previous": "Previous × era factor", "profile_model": "Track-profile model", "method": "How"})
    st.dataframe(show.style.format({c: "{:.2f}" for c in ["Estimate (passes/lap)", "Seen under these rules", "Previous rules",
                                                          "Previous × era factor", "Track-profile model"]}
                                   | {"Upcoming round": "{:.0f}"}, na_rep="–"), hide_index=True, width="stretch")
    v = rep.get("validation", {})
    if v.get("model", {}).get("n"):
        ui.note(f"Walk-forward over the {v['model']['n']} races under the current rules where every method can predict "
                f"(each from earlier races only): error {v['model']['mae_per_lap']:.3f} passes per lap for this estimate, "
                f"{v['previous_era_only']['mae_per_lap']:.3f} for 'previous rules only' and "
                f"{v['last_visit']['mae_per_lap']:.3f} for 'same as last visit'. Track-level passing under the new rules is "
                "close to the previous era's; the era factor adds little at a single track.")


# ------------------------------------------------------------------ 2026 season: predicting winners

SIGNAL_LABEL = {"quali_pos": "Qualifying position", "quali_gap": "Qualifying gap", "fp_best": "Practice best lap",
                "fp_longrun": "Practice long run", "sprint_pace": "Sprint race pace", "profile_pace": "Driver + car profile",
                "v1_model": "Model v1", "v2_model": "Model v2 (with race simulator)"}


def season_page():
    p = ROOT / "reports" / "analysis_2026.json"
    if not p.exists():
        st.info("Run `scripts/analysis_2026.py` first.")
        return
    a = __import__("json").loads(p.read_text())
    sig = a["signals"]["2026"]
    best = max(sig, key=lambda k: (sig[k]["top1"], -sig[k]["mean_rank"]))
    c = a["confidence_2026"]
    ui.title(f"{a['season']} season: predicting the winner",
             f"{SIGNAL_LABEL[best]} picked {sig[best]['top1']:.0%} of the {a['races']} winners. When the confidence model "
             f"gave the favourite 60% or more ({c['high_conf_races']} races) the favourite won "
             f"{(c['high_conf_hit_rate'] or 0):.0%} of the time.", scope="Explore · 2026 analysis")
    wins = a["wins_by_team_2026"]
    ui.kpis([("Races", a["races"]), ("Pole won the race", f"{a['pole_conversion'].get('2026', 0):.0%}", None,
              f"2022–25: {np.mean([a['pole_conversion'][s] for s in ('2022', '2023', '2024', '2025')]):.0%}"),
             ("Most wins", f"{max(wins, key=wins.get).replace('_', ' ').title()} {max(wins.values())}"),
             ("Favourite won", f"{c['fav_hit_rate']:.0%}"),
             ("Right teammate picked", f"{a['teammates']['signal_picked_right'].get('v2_model', 0):.0%}", None,
              f"inside {a['teammates']['team'].title()}")])
    l, r = st.columns(2, gap="medium")
    with l:
        st.subheader("Which pre-race signal finds the winner")
        t = pd.DataFrame([{"signal": SIGNAL_LABEL[k], "Winner was its #1": v["top1"], "In its top 3": v["top3"],
                           "races": v["races"]} for k, v in sig.items()]).sort_values("Winner was its #1", ascending=False)
        st.altair_chart(alt.Chart(t).mark_bar().encode(
            y=alt.Y("signal:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
            x=alt.X("Winner was its #1:Q", axis=alt.Axis(format="%"), title="share of 2026 races its top pick won"),
            color=alt.condition(f"datum.signal == '{SIGNAL_LABEL[best]}'", alt.value(ui.ACCENT), alt.value(ui.QUIET)),
            tooltip=["signal", alt.Tooltip("Winner was its #1:Q", format=".0%"), alt.Tooltip("In its top 3:Q", format=".0%"),
                     "races"]).properties(height=260), width="stretch")
    with r:
        st.subheader("How sure can we be?")
        tiers = pd.DataFrame(a["confidence_tiers"])
        st.altair_chart(alt.Chart(tiers).mark_bar().encode(
            x=alt.X("tier:N", sort=["<40%", "40-60%", "60-75%", ">75%"], title="confidence model: P(favourite wins)"),
            y=alt.Y("fav_won:Q", axis=alt.Axis(format="%"), title="favourite actually won"),
            tooltip=["tier", "races", alt.Tooltip("predicted:Q", format=".0%"), alt.Tooltip("fav_won:Q", format=".0%")])
            .properties(height=260), width="stretch")
        ui.note("Walk-forward over 2023 onward: each race's confidence from a model trained on earlier races only. "
                f"On 2026 favourites its accuracy score (Brier) is {c['brier_conf_model']:.3f} vs {c['brier_v2_pwin']:.3f} "
                "for the raw model probability (lower is better).")
    st.subheader("What makes a favourite safe")
    co = pd.DataFrame({"factor": list(a["confidence_coef_per_sd"]), "effect": list(a["confidence_coef_per_sd"].values())})
    co["factor"] = co["factor"].map({"fav_pwin": "model's win probability", "pole_margin": "pole margin",
                                     "team_margin": "gap from best to second-best team", "fav_start": "favourite's grid slot",
                                     "fav_team_mate_gap": "gap to own teammate", "passing": "ease of passing",
                                     "sc_rate": "safety-car rate"})
    st.altair_chart(alt.Chart(co).mark_bar().encode(
        y=alt.Y("factor:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("effect:Q", title="effect on the favourite winning (log-odds per standard deviation)"),
        color=alt.condition("datum.effect > 0", alt.value(ui.ACCENT), alt.value(ui.QUIET))).properties(height=220),
        width="stretch")
    st.subheader("Race by race")
    races = pd.DataFrame(a["races_2026"])
    races["result"] = np.where(races["fav_won"], "favourite won",
                               "won by " + races["winner"].astype(str) + " from P" + races["winner_start"].astype("Int64").astype(str))
    show = races[["round", "circuit_id", "fav_code", "fav_pwin", "p_fav_wins", "pole_margin", "regime_name", "result"]].rename(columns={
        "round": "Rd", "circuit_id": "Track", "fav_code": "Favourite", "fav_pwin": "Model P(win)",
        "p_fav_wins": "Confidence model", "pole_margin": "Pole margin (%)", "regime_name": "Regime", "result": "Result"})
    st.dataframe(show.style.format({"Model P(win)": "{:.0%}", "Confidence model": "{:.0%}", "Pole margin (%)": "{:.2f}"},
                                   na_rep="–"), hide_index=True, width="stretch")
    shap_cols = [c for c in races.columns if c.startswith("shap_")]
    if shap_cols:
        st.subheader("Why the confidence model said what it said (SHAP)")
        pick = st.selectbox("Race", list(races["round"]), format_func=lambda r: f"R{r} · "
                            f"{races.loc[races['round'] == r, 'circuit_id'].iat[0]}", key="se_shap")
        row = races[races["round"] == pick].iloc[0]
        sv = pd.DataFrame({"factor": [c[5:] for c in shap_cols], "effect": [row[c] for c in shap_cols]})
        sv["factor"] = sv["factor"].map({"fav_pwin": "model's win probability", "pole_margin": "pole margin",
                                         "team_margin": "gap from best to second-best team",
                                         "fav_start": "favourite's grid slot", "fav_team_mate_gap": "gap to own teammate",
                                         "passing": "ease of passing", "sc_rate": "safety-car rate"})
        st.altair_chart(alt.Chart(sv).mark_bar().encode(
            y=alt.Y("factor:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("effect:Q", title="push on 'favourite wins' (log-odds; right = more confident)"),
            color=alt.condition("datum.effect > 0", alt.value(ui.ACCENT), alt.value(ui.QUIET)),
            tooltip=["factor", alt.Tooltip("effect:Q", format="+.2f")]).properties(height=220), width="stretch")
        ui.note(f"{row['fav_code']} at {row['circuit_id']}: confidence {row['p_fav_wins']:.0%}; "
                + ("the favourite won." if row["fav_won"] else f"won by {row['winner']}."))
    ui.note(f"Regimes: k-means on each race's pre-race shape (k={a['regimes']['k']}, silhouette {a['regimes']['silhouette']}). "
            "A low silhouette means 2026 races mostly look alike before the start: one dominant car, close fights behind.")
