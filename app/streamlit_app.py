"""F1 Forecaster dashboard. Reads only files committed in the repo."""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

from eval.metrics import calibration_table  # noqa: E402
import analytics_pages as AP  # noqa: E402
import explore_pages as EP  # noqa: E402
import search  # noqa: E402
import unsupervised_page as UP  # noqa: E402
import ui  # noqa: E402

st.set_page_config(page_title="F1 Forecaster", layout="wide")
DISCLAIMER = "Probabilistic forecasts for analysis only. Not betting or financial advice."


def jsonl(path):
    p = ROOT / path
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def jfile(path, default=None):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else default


def pct(df, cols):
    return df.style.format({c: "{:.1%}" for c in cols} | {"exp_pos": "{:.1f}"})


# ------------------------------------------------------------------ pages

def next_race():
    log = [e for e in jsonl("predictions/log.jsonl") if e["role"] == "champion"]
    if not log:
        st.info("No forecast logged yet.")
        return
    e = log[-1]
    ui.title(f"{e['season']} R{e['round']} · {e['race_name']}", scope="Logged pre-race forecast")
    st.caption(f"Model **{e['model_version']}** · logged {e['logged_at']} · race start {e['race_utc']}"
               f" · commit {e['git_commit']} · feature hash {e['feature_hash']}" + (" · LATE (not counted)" if e["late"] else ""))
    df = pd.DataFrame(e["drivers"]).sort_values("pred_rank")
    fav = df.iloc[0]
    st.markdown(f"Most likely winner: **{fav.code}** at **{fav.p_win:.0%}**, so a {1 - fav.p_win:.0%} chance someone else wins.")
    c = alt.Chart(df).mark_bar().encode(x=alt.X("p_win:Q", axis=alt.Axis(format="%"), title="P(win)"),
                                        y=alt.Y("code:N", sort="-x", title=None), tooltip=["code", "p_win"])
    st.altair_chart(c, width="stretch")
    show = df[["pred_rank", "code", "constructor_id", "start_pos", "p_win", "p_top3", "p_top10", "p_dnf", "exp_pos"]]
    st.dataframe(pct(show, ["p_win", "p_top3", "p_top10", "p_dnf"]), hide_index=True, width="stretch")
    st.caption("All probabilities come from one simulation of 10,000 races, so they are mutually consistent.")


def last_weekend():
    reports = sorted((ROOT / "reports").glob("*_pace.md"))
    if not reports:
        st.info("No post-race review yet.")
        return
    pick = st.selectbox("Race", [r.stem for r in reversed(reports)])
    st.markdown((ROOT / "reports" / f"{pick}.md").read_text())


def championship():
    files = sorted((ROOT / "data" / "championship").glob("*.json"))
    if not files:
        st.info("No championship simulation yet.")
        return
    season = st.selectbox("Season", [f.stem for f in reversed(files)])
    hist = jfile(f"data/championship/{season}.json")
    last = hist[max(hist, key=int)]
    st.caption(f"After round {last['after_round']}, {last['remaining']} races left, model {last['model_version']}. "
               "Assumes current line-ups; fastest-lap points not simulated.")
    for kind, key in [("Drivers", "drivers"), ("Constructors", "constructors")]:
        st.subheader(kind)
        df = pd.DataFrame(last[key])
        name = "code" if key == "drivers" else "constructor_id"
        st.dataframe(pct(df[[name, "points", "exp_points", "p_title"]].head(12), ["p_title"])
                     .format({"exp_points": "{:.0f}", "points": "{:.0f}"}), hide_index=True, width="stretch")
        rows = [{"round": int(r), name: d[name], "p_title": d["p_title"]} for r, h in hist.items() for d in h[key]]
        t = pd.DataFrame(rows)
        top = t.groupby(name)["p_title"].max().nlargest(6).index
        if t["round"].nunique() > 1:
            st.altair_chart(alt.Chart(t[t[name].isin(top)]).mark_line(point=True).encode(
                x="round:Q", y=alt.Y("p_title:Q", axis=alt.Axis(format="%")), color=f"{name}:N"),
                width="stretch")


def scorecard():
    scores = [s for s in jsonl("predictions/scores.jsonl") if s["role"] == "champion" and not s["late"]]
    st.subheader("Live record (pre-race logged forecasts only)")
    if scores:
        rows = []
        for s in scores:
            for who, m in [("model", s["metrics"])] + list(s["baselines"].items()):
                rows.append({"race": f"{s['season']} R{s['round']}", "who": who, **m})
        df = pd.DataFrame(rows)
        agg = df.groupby("who")[["winner_logloss", "podium_brier", "top10_brier", "spearman", "top1_hit", "mae_pos"]].mean()
        st.dataframe(agg.style.format("{:.3f}"), width="stretch")
        n = len(scores)
        st.caption(f"{n} live race(s). With fewer than ~20 races these numbers are very noisy.")
        st.dataframe(df[df.who == "model"].drop(columns="who"), hide_index=True, width="stretch")
    else:
        st.info("No live race scored yet.")
    st.subheader("Walk-forward backtest (guides development; not live accuracy)")
    bt = jfile("reports/backtest.json")
    if bt:
        o = pd.DataFrame(bt["overall"]).T
        cols = ["n_races", "winner_logloss", "podium_brier", "top10_brier", "spearman", "top1_hit", "ece_win", "ece_top3", "ece_top10"]
        st.dataframe(o[cols].astype(float).style.format("{:.3f}"), width="stretch")
        ci = bt["overall"]["model"]
        st.caption(f"Model winner log-loss 95% CI {ci['winner_logloss_ci'][0]:.3f} to {ci['winner_logloss_ci'][1]:.3f}; "
                   f"Spearman CI {ci['spearman_ci'][0]:.3f} to {ci['spearman_ci'][1]:.3f} (bootstrap over races).")
        bs = pd.DataFrame(bt["by_season"]["model"]).T.add_prefix("model_").join(
            pd.DataFrame(bt["by_season"]["grid"]).T.add_prefix("grid_"))
        st.dataframe(bs[["model_winner_logloss", "grid_winner_logloss", "model_spearman", "grid_spearman"]]
                     .style.format("{:.3f}"), width="stretch")
    preds = [pd.DataFrame(e["drivers"]).assign(season=e["season"], round=e["round"])
             for e in jsonl("predictions/log.jsonl") if e["role"] == "champion" and not e["late"]]
    res_files = sorted((ROOT / "data" / "results").glob("*.csv"))
    if preds and res_files:
        res = pd.concat([pd.read_csv(f) for f in res_files[-2:]])
        p = pd.concat(preds).merge(res[["season", "round", "driver_id", "position"]], on=["season", "round", "driver_id"])
        if len(p):
            st.subheader("Live calibration (top-10)")
            st.dataframe(calibration_table(p.p_top10, p.position <= 10), hide_index=True)


def model_log():
    reg = jfile("models/registry.json", {})
    if reg:
        st.markdown(f"**Champion:** {reg['champion']} · **Shadow:** {reg.get('shadow') or 'none'}"
                    + (f" (since {reg['shadow_since']})" if reg.get("shadow_since") else ""))
        st.json(reg["versions"][reg["champion"]]["spec"], expanded=False)
        st.dataframe(pd.DataFrame(reg["history"]), hide_index=True, width="stretch")
    exps = jsonl("experiments/log.jsonl")
    cycles = [e for e in exps if e.get("event") == "cycle"]
    if cycles:
        st.info(cycles[-1]["nn_readiness"])
    st.subheader("Experiments")
    rows = [{"date": e["date"], "slug": e["slug"], "outcome": e.get("outcome"),
             "composite": e["decision"].get("composite"), "bootstrap": e["decision"].get("bootstrap_win_rate"),
             "hypothesis": e.get("hypothesis")} for e in exps if "decision" in e]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.subheader("Pending proposals (need a human)")
    props = sorted((ROOT / "proposals").glob("*.md"))
    for p in props:
        with st.expander(p.stem):
            st.markdown(p.read_text())
    if not props:
        st.caption("None.")
    cl = ROOT / "CHANGELOG_MODELS.md"
    if cl.exists():
        with st.expander("Changelog"):
            st.markdown(cl.read_text())


PAGES = {
    "": st.Page(AP.projections, title="Next race", url_path="projections", default=True),
    "weekend": st.Page(AP.weekend_lab, title="Weekend lab", url_path="weekend"),
    "race": st.Page(AP.race_story, title="Race story", url_path="race"),
    "next": st.Page(next_race, title="Logged forecast", url_path="next"),
    "last": st.Page(last_weekend, title="Post-race review", url_path="last"),
    "driver": st.Page(EP.driver_page, title="Drivers", url_path="driver"),
    "constructor": st.Page(EP.constructor_page, title="Constructors", url_path="constructor"),
    "track": st.Page(EP.track_page, title="Tracks", url_path="track"),
    "teams": st.Page(AP.team_dna, title="Team DNA", url_path="teams"),
    "drivers": st.Page(AP.driver_dna, title="Driver DNA", url_path="drivers"),
    "circuits": st.Page(AP.circuits, title="Circuit types", url_path="circuits"),
    "eras": st.Page(EP.eras_page, title="Regulation eras", url_path="eras"),
    "season": st.Page(EP.season_page, title="2026: predicting winners", url_path="season"),
    "unsupervised": st.Page(UP.unsupervised_page, title="Unsupervised lab", url_path="unsupervised"),
    "championship": st.Page(championship, title="Championship", url_path="championship"),
    "ml": st.Page(AP.ml_lab, title="ML lab", url_path="ml"),
    "scorecard": st.Page(scorecard, title="Scorecard", url_path="scorecard"),
    "models": st.Page(model_log, title="Model log", url_path="models"),
}
SECTIONS = {"Race weekend": ["", "weekend", "race", "next", "last"],
            "Explore": ["season", "unsupervised", "driver", "constructor", "track", "teams", "drivers", "circuits", "eras", "championship"],
            "Models": ["ml", "scorecard", "models"]}
pg = st.navigation({k: [PAGES[u] for u in v] for k, v in SECTIONS.items()}, expanded=True)
search.maybe_switch(PAGES)
ui.setup()
ui.help_button(pg.url_path)
search.box()
_reg = jfile("models/registry.json", {})
ui.brand(f"Champion {_reg.get('champion', '–')} · 2014–2026 results, 2023+ telemetry")
st.sidebar.caption(DISCLAIMER)
pg.run()
