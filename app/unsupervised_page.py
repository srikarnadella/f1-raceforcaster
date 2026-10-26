"""Unsupervised lab: track affinity, driver archetypes, car character, circuit types, anomalies."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ui
from analytics_pages import read, read_json, team_scale
from explore_pages import TRAIT_LABEL, results


def _current_drivers():
    r = results()
    last = int(r["season"].max())
    return list(r[r["season"] == last].groupby("driver_id")["points"].sum().sort_values(ascending=False).index)


def unsupervised_page():
    rep = read_json("report").get("affinity", {})
    tmp = rep.get("temporal", {})
    ct = tmp.get("same circuit type", {})
    sc = tmp.get("same circuit", {})
    ui.title("Unsupervised lab",
             "Patterns found without labels: where drivers are unusually strong, driver and car types, circuit "
             f"groups and odd weekends. Track affinity is a weak signal (correlation {sc.get('correlation', 0):.2f} "
             f"with the next visit at the same track, {ct.get('correlation', 0):.2f} at the same circuit type), so treat "
             "single-track numbers as history, not prediction.", scope="Explore · unsupervised learning")
    tabs = st.tabs(["Track affinity", "Driver × track type", "Latent map", "Driver archetypes", "Car character",
                    "Anomalies"])
    cells = read("affinity_cells")
    cur = _current_drivers()
    names = cells.drop_duplicates("driver_id").set_index("driver_id")["code"].to_dict() if len(cells) else {}

    with tabs[0]:
        if cells.empty:
            st.info("Run `scripts/build_analytics.py`.")
        else:
            l, r = st.columns(2, gap="medium")
            with l:
                opts = [d for d in cur if d in names] + [d for d in names if d not in cur]
                drv = st.selectbox("Driver", opts, format_func=lambda d: f"{names.get(d, d)} · {d.replace('_', ' ').title()}",
                                   key="ul_drv")
                c = cells[cells["driver_id"] == drv].copy()
                c["signal"] = np.where((c["t"].abs() >= 2) & (c["count"] >= 3), "notable (|t| ≥ 2, 3+ visits)", "within noise")
                c["track"] = c["circuit_id"].str.replace("_", " ")
                st.altair_chart(alt.Chart(c).mark_bar().encode(
                    y=alt.Y("track:N", sort="x", title=None, axis=alt.Axis(labelOverlap=False)),
                    x=alt.X("affinity:Q", title="qualifying vs teammate, vs own usual level (%, left = stronger)"),
                    color=alt.Color("signal:N", scale=alt.Scale(domain=["notable (|t| ≥ 2, 3+ visits)", "within noise"],
                                                                range=[ui.ACCENT, ui.QUIET]), title=None),
                    tooltip=["track", "circuit_type", "count", alt.Tooltip("raw:Q", format="+.3f"),
                             alt.Tooltip("affinity:Q", format="+.3f"), alt.Tooltip("t:Q", format="+.1f")])
                    .properties(height=alt.Step(18)), width="stretch")
            with r:
                circ = st.selectbox("Who is strongest at this track (current drivers)", sorted(cells["circuit_id"].unique()),
                                    format_func=lambda x: x.replace("_", " "), key="ul_circ")
                w = cells[(cells["circuit_id"] == circ) & cells["driver_id"].isin(cur)].sort_values("affinity").copy()
                w["t"] = pd.to_numeric(w["t"], errors="coerce")
                st.dataframe(w[["code", "count", "affinity", "t", "circuit_type"]].rename(columns={
                    "code": "Driver", "count": "Visits", "affinity": "Affinity (%)", "t": "t-stat", "circuit_type": "Type"})
                    .style.format({"Affinity (%)": "{:+.3f}", "t-stat": "{:+.1f}"}, na_rep="–"), hide_index=True,
                    width="stretch", height=420)
            ui.note("Affinity = the driver's qualifying gap to the teammate at this track minus their gap everywhere, shrunk "
                    "toward zero for few visits. Negative = better than their own norm here. 2014 onward.")

    with tabs[1]:
        tt = read("affinity_types")
        if len(tt):
            cols = [c for c in tt.columns if c not in ("driver_id", "track_group", "code")]
            sel = tt[tt["driver_id"].isin(cur)] if tt["driver_id"].isin(cur).any() else tt
            long = sel.melt(id_vars=["driver_id", "code", "track_group"], value_vars=cols, var_name="type", value_name="aff")
            order = list(sel.sort_values("track_group")["code"])
            st.altair_chart(alt.Chart(long).mark_rect().encode(
                x=alt.X("type:N", title=None, axis=alt.Axis(labelAngle=-20, labelLimit=160)),
                y=alt.Y("code:N", sort=order, title=None),
                color=alt.Color("aff:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True, domainMid=0),
                                title="affinity %"),
                tooltip=["code", "type", alt.Tooltip("aff:Q", format="+.3f"), "track_group"]).properties(height=520),
                width="stretch")
            st.markdown(f"Circuit-type affinity is the more reliable level: earlier visits predict the next one with "
                        f"correlation {ct.get('correlation', 0):.2f} (error {ct.get('mae_model', 0):.3f} vs "
                        f"{ct.get('mae_usual_level', 0):.3f} for 'usual level').")
            st.dataframe(sel[["code", "track_group"]].rename(columns={"code": "Driver", "track_group": "Group (k-means)"}),
                         hide_index=True, width="stretch")

    with tabs[2]:
        fd, fc = read("affinity_drivers"), read("affinity_circuits")
        if len(fd):
            fd = fd[fd["driver_id"].isin(cur)] if fd["driver_id"].isin(cur).any() else fd
            dpts = alt.Chart(fd).mark_circle(size=110, color=ui.ACCENT).encode(
                x=alt.X("f1:Q", title="latent factor 1"), y=alt.Y("f2:Q", title="latent factor 2"),
                tooltip=["code", "cocluster"])
            dlab = dpts.mark_text(dy=-10, color=ui.INK, fontWeight=600).encode(text="code:N")
            cpts = alt.Chart(fc).mark_text(color=ui.MUTED, fontSize=10).encode(
                x="f1:Q", y="f2:Q", text="circuit_id:N", tooltip=["circuit_id", "circuit_type", "cocluster"])
            st.altair_chart((cpts + dpts + dlab).properties(height=520), width="stretch")
            h = rep.get("holdout", {})
            ui.note("Matrix factorisation of the driver × track affinity matrix (soft-impute, 2 factors): a driver near a "
                    "track tends to do better there than usual. Honesty check on hidden cells: factor model error "
                    f"{h.get('factor_model', 0):.3f} vs {h.get('usual_level (0)', 0):.3f} for 'usual level', so the map "
                    "describes past patterns; it does not beat the simple baseline at predicting new ones.")

    with tabs[3]:
        lat = read("profiles_latest")
        if len(lat):
            zc = [f"z_{k}" for k in TRAIT_LABEL if f"z_{k}" in lat.columns]
            long = lat.melt(id_vars=["code", "archetype"], value_vars=zc, var_name="trait", value_name="z")
            long["trait"] = long["trait"].str[2:].map(TRAIT_LABEL)
            st.altair_chart(alt.Chart(long.dropna(subset=["z"])).mark_rect().encode(
                x=alt.X("trait:N", sort=list(TRAIT_LABEL.values()), title=None, axis=alt.Axis(labelAngle=-20)),
                y=alt.Y("code:N", sort=list(lat.sort_values("archetype")["code"]), title=None),
                color=alt.Color("z:Q", scale=alt.Scale(scheme="redyellowgreen", domainMid=0), title="vs grid (sd)"),
                tooltip=["code", "trait", alt.Tooltip("z:Q", format="+.2f"), "archetype"]).properties(height=520),
                width="stretch")
            st.dataframe(lat[["code", "archetype", "races_profiled"]].rename(columns={
                "code": "Driver", "archetype": "Archetype (k-means)", "races_profiled": "Races"}), hide_index=True,
                width="stretch")
            ui.note("Each trait is against the teammate, then standardised across the grid; green = better. Drivers with "
                    "fewer than 8 profiled races are left out.")

    with tabs[4]:
        cm = read("car_map")
        if len(cm):
            season = st.selectbox("Season", sorted(cm["season"].unique(), reverse=True), key="ul_cm")
            x = cm[cm["season"] == season]
            pts = alt.Chart(x).mark_circle(size=220).encode(
                x="pc1:Q", y="pc2:Q", color=alt.Color("team_name:N", scale=team_scale(x), legend=None),
                tooltip=["team_name", "z_straight", "z_fast", "z_medium", "z_slow"])
            st.altair_chart((pts + pts.mark_text(dy=-15).encode(text="team_name:N")).properties(height=440),
                            width="stretch")
            ui.note("PCA of where each car gains and loses time (straights, fast, medium, slow corners). Close = similar car.")

    with tabs[5]:
        an = read("anomalies")
        if len(an):
            st.dataframe(an[["season", "round", "circuit_id", "team_name", "quali_gap_pct", "race_pace_gap_pct",
                             "anomaly_score"]].rename(columns={"circuit_id": "track", "team_name": "team",
                                                               "quali_gap_pct": "quali gap %", "race_pace_gap_pct": "race gap %"})
                         .head(25), hide_index=True, width="stretch")
            ui.note("Isolation forest on how far each team's weekend was from its own recent norm. Top rows = strangest weekends.")
