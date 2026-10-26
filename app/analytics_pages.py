"""Analytics pages (OpenF1 era, 2023 onward). Read data/analytics/ and
data/openf1/ only; no API calls from the dashboard."""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import ui
from f1.analytics import telemetry as T
from f1.analytics import weekend as W

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "data" / "analytics"
COMPOUND_COLOURS = {"SOFT": "#E8002D", "MEDIUM": "#FFD12E", "HARD": "#C9C9C9", "INTERMEDIATE": "#43B02A",
                    "WET": "#0067AD"}
ZONE_LABEL = {"straight": "Straights", "fast": "Fast corners", "medium": "Medium corners", "slow": "Slow corners"}
ZONE_HELP = "Fast corners: apex ≥ 210 km/h · medium: 130–210 · slow: < 130 · straights: full throttle, no brake."
ZONES = ["straight", "fast", "medium", "slow"]


@st.cache_data(ttl=600)
def read(name):
    p = A / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=600)
def read_json(name):
    p = A / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(ttl=600)
def load_weekend(season, rnd):
    return W.load(season, rnd)


def team_scale(df, field="team_name", colour="team_colour"):
    m = df[[field, colour]].dropna().drop_duplicates(field)
    return alt.Scale(domain=list(m[field]), range=list(m[colour]))


def need_data():
    if read("weekend").empty:
        st.info("No OpenF1 data built yet. Run `scripts/fetch_openf1.py` then `scripts/build_analytics.py`.")
        return True
    return False


def pick_weekend(key, need=None):
    wk = read("weekend").sort_values(["season", "round"], ascending=False)
    if need is not None:
        wk = wk[wk.apply(need, axis=1)]
    labels = [f"{int(r.season)} R{int(r['round'])} · {r.race_name}" for _, r in wk.iterrows()]
    i = st.selectbox("Weekend", range(len(labels)), format_func=lambda i: labels[i], key=key)
    r = wk.iloc[i]
    return int(r.season), int(r["round"]), r


# ------------------------------------------------------------------ weekend lab

def weekend_lab():
    if need_data():
        return
    head = st.container()
    season, rnd, meta = pick_weekend("wl")
    w = load_weekend(season, rnd)
    laps = w["laps"]
    sessions = [s for s in W.openf1.SESSION_ORDER if s in set(laps["session"])]
    ol_all = W.one_lap(laps)
    bits = []
    q = ol_all[ol_all["session"] == "Q"].sort_values("best_s")
    if len(q) > 1:
        bits.append(f"{q['code'].iat[0]} took pole by {q['best_s'].iat[1] - q['best_s'].iat[0]:.3f} s from {q['code'].iat[1]}")
    for fs in ("FP2", "FP1", "FP3"):
        lr0 = W.long_runs(laps, fs) if fs in sessions else pd.DataFrame()
        if len(lr0):
            top = lr0.sort_values("long_run_s").iloc[0]
            bits.append(f"the best {fs} long run was {top.code}'s on {str(top.main_compound).lower()}s")
            break
    with head:
        ui.title(meta.race_name, (bits[0][0].upper() + "; ".join(bits)[1:] + ".") if bits else None,
                 scope=f"{season} · round {rnd} · {', '.join(sessions)}")
    tabs = st.tabs(["Session pace", "Long runs", "Tyres", "Run programmes", "Telemetry", "Track dominance", "Lap classes"])

    with tabs[0]:
        s = st.radio("Session", sessions, horizontal=True, index=sessions.index("Q") if "Q" in sessions else 0, key="wl_s")
        ol = W.one_lap(laps)
        ol = ol[ol["session"] == s].sort_values("best_s")
        c = alt.Chart(ol).mark_bar().encode(
            x=alt.X("gap_pct:Q", title="gap to fastest (%)", axis=alt.Axis(tickCount=6)),
            y=alt.Y("code:N", sort=list(ol["code"]), title=None),
            color=alt.Color("team_name:N", scale=team_scale(ol), legend=None),
            tooltip=["code", "team_name", alt.Tooltip("best_s:Q", format=".3f"), alt.Tooltip("ideal_s:Q", format=".3f"),
                     "best_compound", "top_speed", "laps"])
        st.altair_chart(c, width="stretch")
        t = ol[["code", "team_name", "best_s", "gap_pct", "ideal_s", "s1", "s2", "s3", "top_speed", "best_compound",
                "laps"]].rename(columns={"code": "Driver", "team_name": "Team", "best_s": "Best lap (s)",
                                         "gap_pct": "Gap (%)", "ideal_s": "Ideal lap (s)", "s1": "S1", "s2": "S2", "s3": "S3",
                                         "top_speed": "Speed trap (km/h)", "best_compound": "Tyre", "laps": "Laps"})
        st.dataframe(t.style.format({"Best lap (s)": "{:.3f}", "Ideal lap (s)": "{:.3f}", "Gap (%)": "{:.2f}", "S1": "{:.3f}",
                                     "S2": "{:.3f}", "S3": "{:.3f}", "Speed trap (km/h)": "{:.0f}"}, na_rep="–"),
                     hide_index=True, width="stretch")
        st.caption("Ideal lap = sum of the driver's best sectors. Top speed = speed trap.")

    with tabs[1]:
        fps = [x for x in sessions if x.startswith("FP")]
        if not fps:
            st.info("No practice sessions this weekend.")
        else:
            s = st.radio("Practice session", fps, horizontal=True, index=len(fps) - 1 if "FP2" not in fps else fps.index("FP2"),
                         key="wl_lr")
            lr = W.long_runs(laps, s)
            if lr.empty:
                st.info("Not enough long-run laps in this session.")
            else:
                lr = lr.sort_values("long_run_s")
                st.markdown("**Long-run pace** corrected for compound and tyre age (fuel load unknown in practice: "
                            "teams run different loads, so treat gaps under ~0.3% as noise).")
                c = alt.Chart(lr).mark_bar().encode(
                    x=alt.X("long_run_gap_pct:Q", title="long-run gap (%)"),
                    y=alt.Y("code:N", sort=list(lr["code"]), title=None),
                    color=alt.Color("team_name:N", scale=team_scale(lr), legend=None),
                    tooltip=["code", "team_name", "long_laps", "main_compound", alt.Tooltip("raw_median_s:Q", format=".3f")])
                st.altair_chart(c, width="stretch")
                st.json(lr.attrs, expanded=False)
                runs = laps[(laps["session"] == s) & (laps["category"] == "long")].copy()
                pick = st.multiselect("Show runs for", list(lr["code"]), default=list(lr["code"][:4]), key="wl_lrd")
                runs = runs[runs["code"].isin(pick)]
                runs["run"] = runs["code"] + " · stint " + runs["stint"].astype("Int64").astype(str) + " · " + runs["compound"].fillna("?")
                st.altair_chart(alt.Chart(runs).mark_line(point=True).encode(
                    x=alt.X("tyre_age:Q", title="tyre age (laps)"),
                    y=alt.Y("lap_s:Q", scale=alt.Scale(zero=False), title="lap time (s)"),
                    color=alt.Color("run:N"), tooltip=["code", "lap", "lap_s", "compound", "tyre_age"]), width="stretch")

    with tabs[2]:
        dg = W.degradation(laps)
        if dg.empty:
            st.info("Not enough laps.")
        else:
            s = st.radio("Session", sorted(set(dg["session"]), key=W.openf1.SESSION_ORDER.index), horizontal=True,
                         index=0, key="wl_dg")
            d = dg[dg["session"] == s]
            st.altair_chart(alt.Chart(d).mark_rect().encode(
                x=alt.X("compound:N", sort=W.COMPOUND_ORDER), y=alt.Y("team_name:N", title=None),
                color=alt.Color("deg_s_per_lap:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True),
                                title="s/lap"),
                tooltip=["team_name", "compound", "deg_s_per_lap", "laps"]), width="stretch")
            st.caption("Lap-time loss per lap of tyre age within a stint"
                       + (" (fuel-corrected)." if s in W.RACE_SESSIONS else ". Practice: fuel not corrected, so read as relative."))

    with tabs[3]:
        rp = read("run_programmes")
        rp = rp[(rp["season"] == season) & (rp["round"] == rnd)] if len(rp) else rp
        if rp.empty:
            st.info("No practice stints.")
        else:
            st.markdown("**Unsupervised run classification.** A Gaussian mixture groups practice stints by length, "
                        "pace relative to the driver's best, lap-time spread and tyre age. Clusters are named from their "
                        "centres: longest = race sim, fastest of the rest = quali sim.")
            st.altair_chart(alt.Chart(rp).mark_circle(size=70).encode(
                x=alt.X("n:Q", title="laps in run", scale=alt.Scale(type="log")),
                y=alt.Y("pace:Q", title="pace vs driver's best (ratio)", scale=alt.Scale(zero=False)),
                color="programme:N", shape="session:N",
                tooltip=["code", "session", "stint", "n", alt.Tooltip("pace:Q", format=".3f"), "compound", "programme"]),
                width="stretch")
            st.dataframe(rp.pivot_table(index="code", columns="programme", values="n", aggfunc="count", fill_value=0),
                         width="stretch")

    with tabs[4]:
        tel = T.load(season, rnd)
        if tel is None:
            st.info("No qualifying telemetry for this weekend.")
        else:
            ids = w["drivers"].set_index("driver_number")
            order = tel.drop_duplicates("driver_number").sort_values("lap_s")
            opts = {ids["code"].get(n, str(n)): int(n) for n in order["driver_number"]}
            pick = st.multiselect("Drivers (fastest quali laps)", list(opts), default=list(opts)[:2], key="wl_tel")
            tr = T.speed_traces(season, rnd, [opts[p] for p in pick])
            if len(tr):
                tr["code"] = tr["driver_number"].map(ids["code"])
                base = alt.Chart(tr).encode(x=alt.X("dist:Q", title="distance (m)"), color="code:N")
                st.altair_chart(base.mark_line().encode(y=alt.Y("speed:Q", title="speed (km/h)")).properties(height=260),
                                width="stretch")
                st.altair_chart(base.mark_line().encode(y=alt.Y("delta_s:Q", title="time vs fastest (s)")).properties(height=160),
                                width="stretch")
                st.altair_chart(base.mark_line().encode(y=alt.Y("throttle:Q", title="throttle %")).properties(height=120),
                                width="stretch")
            zt, cls, z = T.zone_table(tel, w["drivers"])
            st.markdown("**Where the lap time went** (vs pole, per zone class; adds up to the gap to pole)")
            c2 = cls.copy()
            c2["zone"] = c2["kind"].map(ZONE_LABEL)
            order_codes = list(c2.groupby("code")["delta_to_pole_s"].sum().sort_values().index)
            st.altair_chart(alt.Chart(c2).mark_bar().encode(
                y=alt.Y("code:N", sort=order_codes, title=None), x=alt.X("delta_to_pole_s:Q", title="seconds vs pole"),
                color=alt.Color("zone:N", scale=alt.Scale(domain=[ZONE_LABEL[k] for k in ZONES],
                                                          range=["#4C78A8", "#72B7B2", "#F58518", "#E45756"])),
                tooltip=["code", "zone", alt.Tooltip("delta_to_pole_s:Q", format=".3f"),
                         alt.Tooltip("speed_vs_median_pct:Q", format=".2f")]), width="stretch")
            prof = T.circuit_profile(tel)
            cc = st.columns(4)
            for i, k in enumerate(ZONES):
                cc[i].metric(ZONE_LABEL[k], f"{prof[f'share_{k}']:.0%} of lap")
            with st.expander("Corner-by-corner apex speeds"):
                cz = zt[zt["kind"] != "straight"]
                st.altair_chart(alt.Chart(cz[cz["code"].isin(pick)] if pick else cz).mark_line(point=True).encode(
                    x=alt.X("corner:O"), y=alt.Y("apex_delta_kmh:Q", title="apex speed vs pole (km/h)"), color="code:N",
                    tooltip=["code", "corner", "kind", "min_speed", "apex_delta_kmh"]), width="stretch")

    with tabs[5]:
        m = T.dominance_map(season, rnd, w["drivers"])
        if m is None:
            st.info("No track map for this weekend.")
        else:
            st.markdown("**Track dominance**: each of 30 mini-sectors coloured by the team whose fastest qualifying "
                        "lap was quickest through it.")
            st.altair_chart(alt.Chart(m).mark_circle(size=60).encode(
                x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
                color=alt.Color("team_name:N", scale=team_scale(m), title="fastest team"),
                tooltip=["sector", "code", "team_name"]).properties(height=520), width="stretch")
            st.dataframe(m.drop_duplicates("sector")["team_name"].value_counts().rename("mini-sectors won"), width="stretch")

    with tabs[6]:
        st.markdown(W.__doc__.split("\n\n", 1)[1].replace("\n  ", "\n- "))
        st.dataframe(laps.pivot_table(index="session", columns="category", values="lap", aggfunc="count", fill_value=0)
                     .reindex(sessions), width="stretch")


# ------------------------------------------------------------------ race story

def race_story():
    if need_data():
        return
    head = st.container()
    season, rnd, meta = pick_weekend("rs")
    w = load_weekend(season, rnd)
    laps = w["laps"]
    if "R" not in set(laps["session"]):
        st.info("No race data for this weekend yet.")
        return
    story = W.race_story(laps)
    last_lap = story[story["lap"] == story["lap"].max()].sort_values("position")
    fin = None
    if len(last_lap) > 1:
        fin = (f"{last_lap['code'].iat[0]} won by {last_lap['gap_to_leader_s'].iat[1]:.1f} s from {last_lap['code'].iat[1]} "
               f"over {int(story['lap'].max())} laps.")
    with head:
        ui.title(meta.race_name, fin, scope=f"{season} · round {rnd} · Race story")
    ses = w["sessions"]
    r = ses[ses["session"] == "R"]
    if len(r):
        r = r.iloc[0]
        ui.kpis([("Track temperature", f"{r.track_temp:.0f}", "°C"), ("Air temperature", f"{r.air_temp:.0f}", "°C"),
                 ("Rain", f"{r.rain_share:.0%}", None, "of weather samples"), ("Safety cars", int(r.sc)),
                 ("Virtual safety cars", int(r.vsc)), ("Red flags", int(r.red_flags))])
    scale = team_scale(story.merge(w["drivers"][["code", "team_name"]], on="code"), "code", "team_colour")
    order_fin = list(story[story["lap"] == story["lap"].max()].sort_values("position")["code"])
    rest = [c for c in story.drop_duplicates("code")["code"] if c not in order_fin]
    hi = st.multiselect("Highlight", order_fin + rest, default=order_fin[:3], key="rs_hi")
    story = story.assign(hl=story["code"].isin(hi))
    ends = story.sort_values("lap").groupby("code").tail(1)

    def lines(y, ytitle, ydomain=None, rev=True):
        sc = alt.Scale(reverse=rev, domain=ydomain) if ydomain else alt.Scale(reverse=rev)
        base = alt.Chart(story).encode(x=alt.X("lap:Q", title="lap"), y=alt.Y(f"{y}:Q", scale=sc, title=ytitle),
                                       detail="code:N", tooltip=["code", "lap", "position", alt.Tooltip("gap_to_leader_s:Q", format=".1f"),
                                                                 "compound"])
        bg = base.transform_filter("!datum.hl").mark_line(strokeWidth=1, color=ui.QUIET)
        fg = base.transform_filter("datum.hl").mark_line(strokeWidth=2.5).encode(color=alt.Color("code:N", scale=scale, legend=None))
        lab = alt.Chart(ends[ends["hl"]]).mark_text(align="left", dx=6, fontSize=12, fontWeight=600).encode(
            x="lap:Q", y=alt.Y(f"{y}:Q", scale=sc), text="code:N", color=alt.Color("code:N", scale=scale, legend=None))
        return (bg + fg + lab).properties(height=360)
    st.subheader("Lap chart")
    st.altair_chart(lines("position", "position", [1, 22]), width="stretch")
    st.subheader("Gap to leader")
    story = story[story["gap_to_leader_s"] < 120]
    ends = ends[ends["gap_to_leader_s"] < 120]
    st.altair_chart(lines("gap_to_leader_s", "seconds behind the leader"), width="stretch")
    st.subheader("Tyre strategy")
    stn = W.stints(laps)
    order = order_fin + [c for c in stn["code"].unique() if c not in order_fin]
    stn["lap_end1"] = stn["lap_end"] + 1
    st.altair_chart(alt.Chart(stn).mark_bar(stroke="#0f1012", strokeWidth=0.5).encode(
        x=alt.X("lap_start:Q", title="lap"), x2="lap_end1:Q",
        y=alt.Y("code:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False)),
        color=alt.Color("compound:N", scale=alt.Scale(domain=list(COMPOUND_COLOURS), range=list(COMPOUND_COLOURS.values()))),
        tooltip=["code", "stint", "compound", "lap_start", "lap_end", "tyre_age_start"]).properties(height=460), width="stretch")
    rp = W.race_pace(laps)
    if len(rp):
        st.subheader("Race pace (clean air, fuel and tyre corrected)")
        rp = rp.sort_values("race_pace_gap_pct")
        st.altair_chart(alt.Chart(rp.dropna(subset=["race_pace_gap_pct"])).mark_bar().encode(
            x=alt.X("race_pace_gap_pct:Q", title="gap to best (%)"), y=alt.Y("code:N", sort=list(rp["code"]), title=None),
            color=alt.Color("team_name:N", scale=team_scale(rp), legend=None),
            tooltip=["code", "clean_laps", "green_laps", alt.Tooltip("race_pace_gap_pct:Q", format=".2f")]), width="stretch")
        deg = rp.attrs.get("deg_s_per_lap", {})
        off = rp.attrs.get("compound_offset_s", {})
        ui.note(f"Fitted on clean-air laps: fuel burn {rp.attrs.get('fuel_s_per_lap', 0):+.3f} s per lap; tyre wear "
                + ", ".join(f"{k.lower()} {v:.3f} s/lap" for k, v in deg.items()) + "; compound pace vs soft "
                + ", ".join(f"{k.lower()} {v:+.2f} s" for k, v in off.items() if k != "SOFT") + ".")
    rc = w["race_control"]
    rc = rc[(rc["session"] == "R") & rc["message"].str.contains("SAFETY CAR|RED FLAG|PENALTY|RETIRED", na=False)]
    if len(rc):
        with st.expander(f"Race control ({len(rc)} key messages)"):
            st.dataframe(rc[["lap_number", "message"]], hide_index=True, width="stretch")


# ------------------------------------------------------------------ team DNA

def team_dna():
    if need_data():
        return
    head = st.container()
    tw = read("team_weekend")
    counts = tw.groupby("season")["round"].nunique()
    seasons = sorted(counts.index, reverse=True)
    default = next((i for i, x in enumerate(seasons) if counts[x] >= 3), 0)
    season = st.selectbox("Season", seasons, index=default, key="td_s")
    t = tw[tw["season"] == season]
    view = st.radio("View", ["Time (s lost per zone, % vs median)", "Speed (apex / top speed, % vs median)"],
                    horizontal=True, key="td_v")
    pre = "z_" if view.startswith("Time") else "v_"
    cols = [f"{pre}{k}" for k in ZONES]
    last_n = st.slider("Weekends to average (most recent)", 3, int(t["round"].nunique()), min(8, int(t["round"].nunique())),
                       key="td_n")
    recent = t[t["round"].isin(sorted(t["round"].unique())[-last_n:])]
    agg = recent.groupby("constructor_id")[cols + ["quali_gap_pct"]].mean()
    names = t.sort_values(["season", "round"]).groupby("constructor_id")[["team_name", "team_colour"]].last()
    agg = agg.join(names).reset_index()
    agg = agg.sort_values("quali_gap_pct")
    with head:
        lead = agg.iloc[0] if len(agg) else None
        ui.title("Team DNA", (f"{lead.team_name} has been the fastest one-lap car over the last {last_n} weekends; each "
                              "row shows where a car gains or loses time against the field median.") if lead is not None else None,
                 scope=f"{season} · qualifying telemetry, fastest lap per team")
    long = agg.melt(id_vars=["team_name", "quali_gap_pct"], value_vars=cols, var_name="zone", value_name="pct")
    long["zone"] = long["zone"].str[2:].map(ZONE_LABEL)
    # speed: higher is better, flip so colour reads the same as time (green = strength)
    long["strength"] = -long["pct"] if pre == "z_" else long["pct"]
    st.altair_chart(alt.Chart(long).mark_rect().encode(
        x=alt.X("zone:N", sort=[ZONE_LABEL[k] for k in ZONES], title=None),
        y=alt.Y("team_name:N", sort=list(agg["team_name"]), title=None),
        color=alt.Color("strength:Q", scale=alt.Scale(scheme="redyellowgreen", domainMid=0), title="strength"),
        tooltip=["team_name", "zone", alt.Tooltip("pct:Q", format=".2f")]).properties(height=360) +
        alt.Chart(long).mark_text(fontSize=11, fontWeight=500).encode(
            x=alt.X("zone:N", sort=[ZONE_LABEL[k] for k in ZONES]), y=alt.Y("team_name:N", sort=list(agg["team_name"])),
            text=alt.Text("pct:Q", format="+.2f"),
            color=alt.condition("abs(datum.strength) > 2", alt.value("#ffffff"), alt.value("#111214"))), width="stretch")
    ui.note(ZONE_HELP)
    st.markdown("**Signature strengths**")
    lines = []
    for _, r in agg.iterrows():
        vals = {k: r[f"{pre}{k}"] for k in ZONES if pd.notna(r[f"{pre}{k}"])}
        if not vals:
            continue
        score = {k: (-v if pre == "z_" else v) for k, v in vals.items()}
        best, worst = max(score, key=score.get), min(score, key=score.get)
        lines.append(f"- **{r.team_name}** (avg quali gap {r.quali_gap_pct:.2f}%): strongest in "
                     f"{ZONE_LABEL[best].lower()}, weakest in {ZONE_LABEL[worst].lower()}")
    st.markdown("\n".join(lines))
    st.subheader("Qualifying gap over the season")
    tt = t.merge(names[["team_name"]].rename(columns={"team_name": "tn"}), left_on="constructor_id", right_index=True)
    tt = tt.assign(team_name=tt["tn"], team_colour=tt["constructor_id"].map(names["team_colour"]))
    st.altair_chart(alt.Chart(tt).mark_line(point=True).encode(
        x="round:Q", y=alt.Y("quali_gap_pct:Q", scale=alt.Scale(reverse=True, domain=[0, 4], clamp=True),
                             title="gap to best team (%, capped at 4)"),
        color=alt.Color("team_name:N", scale=team_scale(tt)), tooltip=["team_name", "round", "circuit_id",
                                                                       alt.Tooltip("quali_gap_pct:Q", format=".2f")]),
        width="stretch")
    if "race_pace_gap_pct" in t.columns:
        st.subheader("Saturday vs Sunday")
        s2 = t.groupby("constructor_id")[["quali_gap_pct", "race_pace_gap_pct"]].mean().join(names).reset_index()
        pts = alt.Chart(s2).mark_circle(size=160).encode(
            x=alt.X("quali_gap_pct:Q", title="avg quali gap (%)"), y=alt.Y("race_pace_gap_pct:Q", title="avg race pace gap (%)"),
            color=alt.Color("team_name:N", scale=team_scale(s2), legend=None), tooltip=list(s2.columns))
        lab = pts.mark_text(dy=-12).encode(text="team_name:N")
        diag = alt.Chart(pd.DataFrame({"x": [0, s2[["quali_gap_pct", "race_pace_gap_pct"]].max().max()]})).mark_line(
            strokeDash=[4, 4], color="grey").encode(x="x:Q", y="x:Q")
        st.altair_chart(pts + lab + diag, width="stretch")
        st.caption("Below the dashed line = relatively stronger in the race than in qualifying (tyre management, race setup).")
    cm = read("car_map")
    cm = cm[cm["season"] == season] if len(cm) else cm
    if len(cm):
        st.subheader("Car character map (PCA of time-zone DNA)")
        pts = alt.Chart(cm).mark_circle(size=200).encode(
            x="pc1:Q", y="pc2:Q", color=alt.Color("team_name:N", scale=team_scale(cm), legend=None),
            tooltip=["team_name", "z_straight", "z_fast", "z_medium", "z_slow"])
        st.altair_chart(pts + pts.mark_text(dy=-14).encode(text="team_name:N"), width="stretch")
        st.caption("Cars close together lose and gain time in similar places. Axes are combinations of the four zone "
                   "classes (loadings in the ML lab).")


# ------------------------------------------------------------------ driver DNA

def driver_dna():
    dd = read("driver_dna")
    head = st.container()
    if dd.empty:
        st.info("No driver data yet.")
        return
    season = st.selectbox("Season", sorted(dd["season"].unique(), reverse=True), key="dd_s")
    d = dd[(dd["season"] == season) & (dd["weekends"].fillna(0) >= 3)].copy()
    with head:
        best = d.sort_values("tm_quali_pct").iloc[0] if len(d) and "tm_quali_pct" in d else None
        ui.title("Driver DNA", (f"{best.code} has the biggest qualifying margin over a teammate this season "
                                f"({best.tm_quali_pct:+.2f}%). Every figure is against the teammate, so the car cancels out; "
                                "negative = faster.") if best is not None else None, scope=f"{season} · teammate-relative")
    show = ["code", "weekends", "tm_quali_pct", "tm_race_pct", "race_minus_quali", "tm_straight", "tm_fast", "tm_medium",
            "tm_slow", "lap1_gain"]
    show = [c for c in show if c in d.columns]
    st.dataframe(d[show].sort_values("tm_quali_pct").style.format({c: "{:+.2f}" for c in show if c not in ("code", "weekends")})
                 .background_gradient(cmap="RdYlGn_r", subset=[c for c in show if c.startswith("tm_")]),
                 hide_index=True, width="stretch")
    if {"tm_quali_pct", "tm_race_pct"} <= set(d.columns):
        pts = alt.Chart(d.dropna(subset=["tm_quali_pct", "tm_race_pct"])).mark_circle(size=120).encode(
            x=alt.X("tm_quali_pct:Q", title="quali vs teammate (%)"), y=alt.Y("tm_race_pct:Q", title="race pace vs teammate (%)"),
            tooltip=show)
        st.altair_chart(pts + pts.mark_text(dy=-10).encode(text="code:N"), width="stretch")
        st.caption("Bottom-left: beats the teammate on both days. Top-left: one-lap specialist. Bottom-right: Sunday driver.")
    ov = read("overtakes")
    if len(ov):
        o = ov[ov["season"] == season].groupby("driver_id")["overtakes"].agg(["sum", "count"]).reset_index()
        o = o.merge(dd[dd["season"] == season][["driver_id", "code"]], on="driver_id", how="left")
        st.subheader("On-track overtakes (green-flag, excluding pit-stop shuffles)")
        st.altair_chart(alt.Chart(o).mark_bar().encode(x=alt.X("sum:Q", title="passes"), y=alt.Y("code:N", sort="-x", title=None),
                                                       tooltip=["code", "sum"]), width="stretch")
    if "lap1_gain" in d.columns:
        st.subheader("Opening lap: places gained (grid → end of lap 1)")
        st.altair_chart(alt.Chart(d.dropna(subset=["lap1_gain"])).mark_bar().encode(
            x=alt.X("lap1_gain:Q", title="avg places gained"), y=alt.Y("code:N", sort="-x", title=None)), width="stretch")


# ------------------------------------------------------------------ circuits

def circuits():
    cl = read("circuits")
    if cl.empty:
        st.info("No circuit profiles yet.")
        return
    rep = read_json("report").get("circuits", {})
    ui.title("Circuit types", f"{len(cl)} circuits fall into {rep.get('k')} groups by where the lap time is spent, "
             "tyre wear, safety cars and how often cars pass.", scope="Unsupervised clustering of circuit profiles")
    st.caption(f"k-means on standardised circuit profiles, k={rep.get('k')} chosen by silhouette "
               f"({rep.get('silhouette')}). PCA explains {sum(rep.get('explained', [0, 0])):.0%} of the variance in 2-D.")
    pts = alt.Chart(cl).mark_circle(size=160).encode(
        x="pc1:Q", y="pc2:Q", color=alt.Color("cluster_name:N", title="type"),
        tooltip=["race_name", "cluster_name", alt.Tooltip("share_straight:Q", format=".0%"),
                 alt.Tooltip("share_slow:Q", format=".0%"), alt.Tooltip("avg_speed_kmh:Q", format=".0f"),
                 alt.Tooltip("race_deg:Q", format=".3f"), "weekends"])
    st.altair_chart((pts + pts.mark_text(dy=-11, fontSize=10).encode(text="circuit_id:N")).properties(height=480),
                    width="stretch")
    with st.expander("PCA loadings"):
        st.json(rep.get("loadings", {}))
    cols = ["race_name", "cluster_name", "share_straight", "share_fast", "share_medium", "share_slow", "avg_speed_kmh",
            "top_speed_kmh", "full_throttle_pct", "braking_pct", "race_deg", "race_track_temp", "sc_rate",
            "overtakes_per_lap", "pit_loss_s"]
    st.dataframe(cl[[c for c in cols if c in cl.columns]].sort_values("cluster_name").style.format(
        {"share_straight": "{:.0%}", "share_fast": "{:.0%}", "share_medium": "{:.0%}", "share_slow": "{:.0%}",
         "avg_speed_kmh": "{:.0f}", "top_speed_kmh": "{:.0f}", "full_throttle_pct": "{:.0f}", "braking_pct": "{:.0f}",
         "race_deg": "{:.3f}", "race_track_temp": "{:.0f}", "sc_rate": "{:.1f}", "overtakes_per_lap": "{:.2f}",
         "pit_loss_s": "{:.1f}"}), hide_index=True, width="stretch")
    sp = read("strategy")
    if len(sp):
        st.subheader("Projected strategy by circuit")
        names = cl.set_index("circuit_id")["race_name"].to_dict()
        cid = st.selectbox("Circuit", sorted(sp["circuit_id"].unique()), format_func=lambda c: names.get(c, c), key="ci_s")
        st.dataframe(sp[sp["circuit_id"] == cid][["stops", "plan", "delta_s", "pace_only_delta_s"]], hide_index=True,
                     width="stretch")


# ------------------------------------------------------------------ projections

def projections():
    p = read_json("projections")
    if not p:
        st.info("No projection built yet.")
        return
    fav = None
    if p.get("preweekend"):
        d0 = pd.DataFrame(p["preweekend"]["drivers"])
        fav = d0.iloc[0]
    tops = pd.DataFrame(p.get("teams", [])).sort_values("blend_gap") if p.get("teams") else None
    fin = []
    if tops is not None and len(tops):
        fin.append(f"{tops['team_name'].iat[0]} project as the fastest car here")
    if fav is not None:
        fin.append(f"{fav.code} is the pre-weekend favourite at {fav.p_win:.0%}, so a {1 - fav.p_win:.0%} chance someone else wins")
    ui.title(p["race_name"], ("; ".join(fin) + ".") if fin else None,
             scope=f"{p['season']} · round {p['round']} · next race")
    st.caption(f"Race start {p['race_utc']} · format {p['format']} · circuit type: **{p.get('cluster', 'unknown')}**"
               + (f" · similar: {', '.join(p['similar'])}" if p.get("similar") else ""))
    if p.get("teams"):
        t = pd.DataFrame(p["teams"])
        st.markdown(f"**Pre-weekend team pace projection** from each team's zone DNA × this circuit's lap profile "
                    f"(measured {p.get('profile_from')}), blended 50/50 with recent overall qualifying form.")
        prof = p.get("profile", {})
        ui.kpis([(f"{ZONE_LABEL[k]}", f"{prof.get(f'share_{k}', 0):.0%}", None, "of the pole lap") for k in ZONES])
        t = t.sort_values("blend_gap")
        st.altair_chart(alt.Chart(t).mark_bar().encode(
            x=alt.X("blend_gap:Q", title="projected one-lap gap to best team (%)", axis=alt.Axis(tickCount=6)),
            y=alt.Y("team_name:N", sort=list(t["team_name"]), title=None),
            color=alt.Color("team_name:N", scale=team_scale(t), legend=None),
            tooltip=["team_name", alt.Tooltip("blend_gap:Q", format=".2f"), alt.Tooltip("dna_gap:Q", format=".2f")]),
            width="stretch")
        contrib = t.melt(id_vars=["team_name"], value_vars=[f"z_{k}" for k in ZONES], var_name="zone", value_name="z")
        contrib["share"] = contrib["zone"].str[2:].map(lambda k: prof.get(f"share_{k}", 0))
        contrib["effect"] = contrib["z"] * contrib["share"]
        contrib["zone"] = contrib["zone"].str[2:].map(ZONE_LABEL)
        st.markdown("**Why**: contribution of each zone type (team strength × how much of this lap it covers)")
        st.altair_chart(alt.Chart(contrib).mark_bar().encode(
            y=alt.Y("team_name:N", sort=list(t["team_name"]), title=None), x=alt.X("effect:Q", title="% vs field median"),
            color=alt.Color("zone:N", scale=alt.Scale(domain=[ZONE_LABEL[k] for k in ZONES],
                                                      range=["#4C78A8", "#72B7B2", "#F58518", "#E45756"])),
            tooltip=["team_name", "zone", alt.Tooltip("effect:Q", format=".3f")]), width="stretch")
        acc = read_json("report").get("dna_projection", {})
        if acc:
            st.caption("Walk-forward accuracy of this kind of projection (team one-lap gap, earlier weekends only): " +
                       "; ".join(f"{k}: MAE {v['mae_pct']:.2f}%, rank corr {v['rank_spearman']:.2f}" for k, v in acc.items()))
    if p.get("preweekend"):
        pw = p["preweekend"]
        st.subheader("Pre-weekend win probabilities")
        st.caption(f"Before any session: {pw['n_grids']} simulated qualifying sessions (team pace from the projection "
                   f"above ± {pw['sigma_team']:.2f}% measured error, teammate splits from this season), each raced "
                   f"{pw['n_races']} times by champion {pw['model_version']}. Replaced by the logged forecast after qualifying.")
        d = pd.DataFrame(pw["drivers"]).head(12)
        if p.get("teams"):
            tc = pd.DataFrame(p["teams"]).set_index("constructor_id")
            d["team_name"] = d["constructor_id"].map(tc["team_name"]).fillna(d["constructor_id"])
            d["team_colour"] = d["constructor_id"].map(tc["team_colour"]).fillna("#888888")
        else:
            d["team_name"], d["team_colour"] = d["constructor_id"], ui.ACCENT
        st.altair_chart(alt.Chart(d).mark_bar().encode(
            x=alt.X("p_win:Q", axis=alt.Axis(format="%", tickCount=6), title="chance of winning"),
            y=alt.Y("code:N", sort="-x", title=None),
            color=alt.Color("team_name:N", scale=team_scale(d), legend=None),
            tooltip=["code", "constructor_id", alt.Tooltip("p_pole:Q", format=".0%"), alt.Tooltip("p_win:Q", format=".1%"),
                     alt.Tooltip("p_top3:Q", format=".0%")]), width="stretch")
        st.dataframe(d[["code", "team_name", "p_pole", "p_win", "p_top3"]].rename(columns={
            "code": "Driver", "team_name": "Team", "p_pole": "Pole", "p_win": "Win", "p_top3": "Podium"}).style.format(
            {"Pole": "{:.0%}", "Win": "{:.1%}", "Podium": "{:.0%}"}), hide_index=True, width="stretch")
    if p.get("strategy"):
        st.subheader("Strategy projection")
        inp = p.get("strategy_inputs", {})
        circ = p.get("circuit", {})
        ui.kpis([("Pit-stop loss", ui.dash(inp.get("pit_loss"), "{:.1f}"), "s"),
                 ("Green-flag passes per lap", ui.dash(circ.get("overtakes_per_lap"))),
                 ("Passing ease", f"{inp.get('passing_ease', 0):.0%}", None, "1 = as easy as Monza or Spa"),
                 ("Safety cars and VSCs per race", ui.dash(circ.get("sc_rate"), "{:.1f}"))])
        sp = pd.DataFrame(p["strategy"])
        st.dataframe(sp[["stops", "plan", "delta_s", "pace_only_delta_s"]].rename(columns={
            "stops": "Stops", "plan": "Plan (tyre + laps)", "delta_s": "vs best (s, incl. track position)",
            "pace_only_delta_s": "vs best (s, pure pace)"}).style.format(
            {"vs best (s, incl. track position)": "{:.1f}", "vs best (s, pure pace)": "{:.1f}"}),
            hide_index=True, width="stretch")
        deg = inp.get("deg", {})
        st.caption("Linear tyre model from earlier dry visits: degradation " +
                   ", ".join(f"{k.lower()} {v:.3f} s/lap" for k, v in deg.items()) +
                   ". Track-position cost per stop is an explicit assumption (see ML lab for its walk-forward check). "
                   "Safety cars change everything; this is the green-flag baseline.")
    if p.get("sessions_done"):
        st.markdown(f"Sessions completed this weekend: {', '.join(p['sessions_done'])}")
    if p.get("race_pace_projection"):
        st.subheader("Projected Sunday race pace (after qualifying)")
        rp = pd.DataFrame(p["race_pace_projection"])
        st.dataframe(rp[["code", "constructor_id", "quali_gap_pct", "proj_race_pace_gap"]].style.format(
            {"quali_gap_pct": "{:.2f}", "proj_race_pace_gap": "{:.2f}"}), hide_index=True, width="stretch")
    log = [json.loads(l) for l in (ROOT / "predictions" / "log.jsonl").read_text().splitlines() if l.strip()] \
        if (ROOT / "predictions" / "log.jsonl").exists() else []
    mine = [e for e in log if e["season"] == p["season"] and e["round"] == p["round"]]
    if mine:
        st.subheader("Logged win probabilities")
        rows = []
        for e in mine:
            for d in e["drivers"]:
                rows.append({"model": f"{e['role']} {e['model_version']}", "code": d["code"], "p_win": d["p_win"],
                             "p_top3": d["p_top3"]})
        df = pd.DataFrame(rows)
        top = df.groupby("code")["p_win"].max().nlargest(8).index
        st.altair_chart(alt.Chart(df[df["code"].isin(top)]).mark_bar().encode(
            x=alt.X("p_win:Q", axis=alt.Axis(format="%")), y=alt.Y("code:N", sort="-x"), color="model:N",
            yOffset="model:N", tooltip=["model", "code", alt.Tooltip("p_win:Q", format=".1%")]), width="stretch")
    else:
        st.caption("Win probabilities appear here once the forecast is logged after qualifying.")
    st.caption("Projections are estimates with wide uncertainty, for analysis only. Not betting or financial advice.")


# ------------------------------------------------------------------ ML lab

def ml_lab():
    ui.title("ML lab", "How each learned component does on races it never saw (walk-forward), against simple baselines.",
             scope="Models · evaluation")
    wa = ROOT / "reports" / "winner_analysis.json"
    if wa.exists():
        w = json.loads(wa.read_text())
        st.subheader("Why race winners are missed")
        ui.kpis([("Favourite won", f"{w['hit_rate']:.0%}", None, f"{w['races']} races, 2019 onward"),
                 ("Average chance given to the favourite", f"{w['mean_fav_p']:.0%}"),
                 ("Average chance given to the actual winner", f"{w['mean_p_winner']:.0%}")])
        mc = pd.DataFrame({"cause": list(w["miss_causes"]), "races": list(w["miss_causes"].values())})
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.altair_chart(alt.Chart(mc).mark_bar().encode(
                y=alt.Y("cause:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
                x=alt.X("races:Q", title="missed races"), tooltip=["cause", "races"]).properties(height=200), width="stretch")
        with c2:
            cal = pd.DataFrame(w["calibration"])
            st.dataframe(cal.rename(columns={"fav_bucket": "Favourite's chance", "races": "Races",
                                             "predicted": "Predicted", "actual": "Actually won"}).style.format(
                {"Predicted": "{:.0%}", "Actually won": "{:.0%}"}), hide_index=True, width="stretch")
        ui.note(" ".join(w.get("notes", [])))
    sl = ROOT / "reports" / "race_sim_lab.json"
    if sl.exists():
        sj = json.loads(sl.read_text())
        runs = sj.get("runs", {"2023-2023": sj})
        st.subheader("Race simulator: driver profiles racing each other")
        for calib, run in runs.items():
            for window, _ in list(run["test"].values())[0].items():
                rows = {k: v[window] for k, v in run["test"].items()}
                df = pd.DataFrame(rows).T.reindex(columns=["winner_logloss", "top1_hit", "podium_brier", "top10_brier", "spearman"])
                st.markdown(f"Calibrated on **{calib}**, tested on **{window}**")
                st.dataframe(df.astype(float).style.format("{:.3f}").format({"top1_hit": "{:.0%}"})
                             .highlight_min(subset=["winner_logloss", "podium_brier", "top10_brier"], color="#1f3b2c")
                             .highlight_max(subset=["spearman", "top1_hit"], color="#1f3b2c"), width="stretch")
        ui.note("Lap-by-lap simulation: each driver's pace, tyre wear, consistency, starts and attack/defence profile, "
                "overtakes that succeed or fail by pace gap and track, pit stops, safety cars and retirements. "
                "Parameters are tuned only on the calibration seasons.")
    lab = {}
    p = ROOT / "reports" / "model_lab.json"
    if p.exists():
        lab = json.loads(p.read_text())
    if lab:
        st.subheader("Race-winner models (walk-forward)")
        st.caption(lab.get("note", ""))
        for window, rows in lab.get("windows", {}).items():
            st.markdown(f"**Test window {window}**")
            df = pd.DataFrame(rows).T
            st.dataframe(df.astype(float).style.format("{:.3f}").highlight_min(
                subset=[c for c in ["winner_logloss", "podium_brier", "top10_brier"] if c in df.columns], color="#1f3b2c")
                .highlight_max(subset=[c for c in ["spearman", "top1_hit"] if c in df.columns], color="#1f3b2c"),
                width="stretch")
        if lab.get("gate"):
            st.markdown("**Promotion gate** (full frozen 2019+ window)")
            st.json(lab["gate"], expanded=False)
    else:
        st.info("Model comparison not run yet (`scripts/model_lab.py`).")
    mp = ROOT / "reports" / "methods.json"
    if mp.exists():
        mj = json.loads(mp.read_text())
        st.subheader("Ways of measuring pace, compared")
        st.caption(f"{mj['weekends']} weekends. " + mj["note"])
        LABEL = {"raw_median": "Race: median of all laps", "green_median": "Race: median of green laps",
                 "clean_median": "Race: median of clean-air laps", "corrected": "Race: fuel + tyre corrected fit",
                 "quali_position": "Qualifying position", "quali_best": "Qualifying best lap",
                 "quali_ideal": "Qualifying ideal lap (best sectors)", "fp_best": "Practice best lap",
                 "fp_longrun": "Practice long-run pace", "sprint_pace": "Sprint race pace"}
        md = pd.DataFrame([{"Measure": LABEL.get(k, k), "Known": v["group"], "Agrees with result": v["agree_spearman"],
                            "95% CI": f"{v['agree_ci'][0]:.2f}–{v['agree_ci'][1]:.2f}" if v.get("agree_ci") else "–",
                            "Repeats next race": v["repeat_spearman"], "Races": v["races"]}
                           for k, v in mj["methods"].items()])
        st.dataframe(md.style.format({"Agrees with result": "{:.2f}", "Repeats next race": "{:.2f}"}, na_rep="–"),
                     hide_index=True, width="stretch")
        ui.note("Race-pace measures are taken from the race itself, so part of their agreement is mechanical; the "
                "'repeats' column is the fairer test of which one measures the car rather than the day.")
    rep = read_json("report")
    if rep.get("race_pace_model"):
        st.subheader("Race-pace projection (gradient boosting)")
        r = rep["race_pace_model"]
        c = st.columns(2)
        c[0].metric("MAE, boosted model", f"{r['gbm']['mae_pct']:.2f}%",
                    f"{r['gbm']['mae_pct'] - r['quali_gap_baseline']['mae_pct']:+.2f} vs quali gap", delta_color="inverse")
        c[1].metric("Rank correlation", f"{r['gbm']['spearman']:.2f}",
                    f"{r['gbm']['spearman'] - r['quali_gap_baseline']['spearman']:+.2f} vs quali gap")
        chk = read("race_pace_check")
        if len(chk):
            st.altair_chart(alt.Chart(chk).mark_circle(opacity=0.4).encode(
                x=alt.X("pred:Q", title="projected race pace gap (%)"), y=alt.Y("race_pace_gap_pct:Q", title="actual (%)"),
                tooltip=["season", "round", "code", "pred", "race_pace_gap_pct"]), width="stretch")
        sh = read("race_pace_shap")
        if len(sh):
            st.markdown("**What drives the race-pace projection (SHAP)**: each dot is one driver-race; position shows "
                        "how much that input moved the projected race-pace gap (right = slower).")
            imp = sh.groupby("feature")["shap"].apply(lambda v: v.abs().mean()).sort_values(ascending=False)
            st.altair_chart(alt.Chart(sh).mark_tick(opacity=0.35, thickness=1.5).encode(
                y=alt.Y("feature:N", sort=list(imp.index), title=None), x=alt.X("shap:Q", title="SHAP value (% of lap)"),
                tooltip=["season", "round", "code", "feature", alt.Tooltip("value:Q", format=".3f"),
                         alt.Tooltip("shap:Q", format="+.3f")]).properties(height=260), width="stretch")
    if rep.get("dna_projection"):
        st.subheader("Team pace projection from zone DNA")
        st.dataframe(pd.DataFrame(rep["dna_projection"]).T, width="stretch")
        st.caption("proj_gap: DNA × circuit profile only. blend_gap: 50/50 with recent form. season_avg_gap / last_gap: "
                   "naive baselines.")
    if rep.get("strategy"):
        st.subheader("Strategy optimiser: stop count")
        r = rep["strategy"]
        st.markdown(f"Walk-forward over {r['n']} dry races: optimal stop count matched the field's median "
                    f"**{r['exact_model']:.0%}** of the time (MAE {r['mae_model']:.2f} stops) vs "
                    f"{r['exact_last_visit']:.0%} for 'same as last visit' (MAE {r['mae_last_visit']:.2f}).")
        chk = read("strategy_check")
        if len(chk):
            st.dataframe(chk, hide_index=True, width="stretch")
    an = read("anomalies")
    if len(an):
        st.subheader("Anomaly detection (isolation forest)")
        st.caption("Team weekends furthest from that team's own recent norm (quali gap, race pace, long runs).")
        st.dataframe(an[["season", "round", "circuit_id", "team_name", "quali_gap_pct", "race_pace_gap_pct", "anomaly_score"]]
                     .head(15), hide_index=True, width="stretch")
