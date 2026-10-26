"""Header search: every page and tool, plus drivers, constructors and tracks.

Picking an entry stores the target page (and any selection it should open with)
in session state; streamlit_app switches page on the next run."""
import streamlit as st

from f1 import ingest

# (label, page url_path, tab or section the tool lives in)
TOOLS = [
    ("Next race projection", "", "team pace, win chances, strategy"),
    ("Pre-weekend win probabilities", "", "Next race"),
    ("Strategy projection / pit plans", "", "Next race"),
    ("Session pace / best laps / ideal lap", "weekend", "Weekend lab › Session pace"),
    ("Practice long runs", "weekend", "Weekend lab › Long runs"),
    ("Tyre degradation by team", "weekend", "Weekend lab › Tyres"),
    ("Run programmes (race sims, quali sims)", "weekend", "Weekend lab › Run programmes"),
    ("Telemetry speed traces", "weekend", "Weekend lab › Telemetry"),
    ("Time lost per corner type", "weekend", "Weekend lab › Telemetry"),
    ("Track dominance map", "weekend", "Weekend lab › Track dominance"),
    ("Lap chart / positions", "race", "Race story"),
    ("Gap to leader", "race", "Race story"),
    ("Tyre strategy chart", "race", "Race story"),
    ("Race pace (fuel and tyre corrected)", "race", "Race story"),
    ("Logged pre-race forecast", "next", "Logged forecast"),
    ("Post-race review / pace report", "last", "Post-race review"),
    ("Driver head-to-head vs teammate", "driver", "Drivers"),
    ("Constructor season pace", "constructor", "Constructors"),
    ("Track history / winners / poles", "track", "Tracks"),
    ("Team DNA / car strengths", "teams", "Team DNA"),
    ("Car character map", "teams", "Team DNA"),
    ("Saturday vs Sunday pace", "teams", "Team DNA"),
    ("Driver DNA / teammate traits", "drivers", "Driver DNA"),
    ("Overtakes / lap 1 gains", "drivers", "Driver DNA"),
    ("Circuit types / clusters", "circuits", "Circuit types"),
    ("Regulation eras / overtakes by era", "eras", "Regulation eras"),
    ("2026 winner analysis / confidence", "season", "2026: predicting winners"),
    ("Track affinity / driver at each track", "unsupervised", "Unsupervised lab › Track affinity"),
    ("Driver × track type heatmap", "unsupervised", "Unsupervised lab"),
    ("Latent map / matrix factorisation", "unsupervised", "Unsupervised lab"),
    ("Driver archetypes / clusters", "unsupervised", "Unsupervised lab"),
    ("Championship title chances", "championship", "Championship"),
    ("Model comparison / neural net", "ml", "ML lab"),
    ("Ways of measuring pace compared", "ml", "ML lab"),
    ("Live scorecard / accuracy", "scorecard", "Scorecard"),
    ("Model log / experiments / proposals", "models", "Model log"),
]


@st.cache_data(ttl=600)
def entries() -> list[dict]:
    out = [{"label": f"{t}  ·  {where}", "page": p, "set": {}} for t, p, where in TOOLS]
    try:
        r = ingest.load_results()
    except Exception:
        return out
    last = int(r["season"].max())
    cur = r[r["season"] == last]
    for d, code in cur.drop_duplicates("driver_id")[["driver_id", "code"]].itertuples(index=False):
        out.append({"label": f"{code} {d.replace('_', ' ').title()}  ·  driver", "page": "driver",
                    "set": {"dr_season": last, "dr_type": "All circuits", "dr_driver": d}})
    for c in cur["constructor_id"].unique():
        out.append({"label": f"{c.replace('_', ' ').title()}  ·  constructor", "page": "constructor",
                    "set": {"co_season": last, "co_type": "All circuits", "co_team": c}})
    cal = ingest.load_calendar()
    names = cal.sort_values("season").drop_duplicates("circuit_id", keep="last")
    for cid, race in names[["circuit_id", "race_name"]].itertuples(index=False):
        if cid in set(r["circuit_id"]):
            out.append({"label": f"{race} ({cid.replace('_', ' ')})  ·  track", "page": "track", "set": {"tr_c": cid}})
    return out


def _go():
    pick = st.session_state.get("f1search_box")
    if pick is None:
        return
    e = next((e for e in entries() if e["label"] == pick), None)
    st.session_state["f1search_box"] = None
    if e:
        for k, v in e["set"].items():
            st.session_state[k] = v
        st.session_state["_f1goto"] = e["page"]


def box():
    labels = [e["label"] for e in entries()]
    with st.container(key="f1search"):
        st.selectbox("Search", labels, index=None, key="f1search_box", on_change=_go,
                     placeholder="Search tools, drivers, teams, tracks…", label_visibility="collapsed")


def maybe_switch(pages: dict):
    target = st.session_state.pop("_f1goto", None)
    if target is not None and target in pages:
        st.switch_page(pages[target])
