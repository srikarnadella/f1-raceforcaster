"""Shared look for every screen (taste-skill "framed-sheet" style, applied by hand).

- The frame (sidebar + top bar) is one tinted tone; the work sits on a white sheet
  inset 8px with a 1px edge and 16px corners, no shadow.
- Colour goes on data: team colours on marks, the teal accent only on selection.
- KPIs are type in one strip split by hairlines, not tiles.
- Every screen opens with a title row: the title, then the one finding in a sentence.
"""
from __future__ import annotations

import html

import altair as alt
import streamlit as st

INK, MUTED, ACCENT, RULE, BORDER = "#ececee", "#a3a6ad", "#4fb3c8", "#26282c", "#2a2c31"
POSITIVE, NEGATIVE = "#4cc38a", "#f06272"
QUIET = "#3d4046"          # de-emphasised marks (other drivers, the slower side of a split)
SHEET = "#16171a"
FONT = "IBM Plex Sans, Inter, system-ui, sans-serif"
FRAME = "#0f1012"

CSS = f"""
<style>
.stApp {{ background: {FRAME}; }}
header[data-testid="stHeader"] {{ background: {FRAME}; }}
[data-testid="stMain"] {{ background: {SHEET}; border: 1px solid {BORDER}; border-radius: 16px;
  margin: 8px 8px 8px 0; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 56px; padding-bottom: 48px; max-width: 1240px; }}
[data-testid="stSidebarHeader"] {{ padding-bottom: 4px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
h1 {{ font-size: 22px !important; font-weight: 600 !important; padding-bottom: 0 !important; }}
.f1-finding {{ font-size: 15px; color: {INK}; margin: 2px 0 20px; max-width: 72ch; line-height: 1.5; }}
.f1-scope {{ font-size: 13px; color: {MUTED}; margin: 0 0 2px; }}
.f1-kpis {{ display: grid; border: 1px solid {BORDER}; border-radius: 8px; margin: 4px 0 24px; }}
.f1-kpi {{ padding: 14px 18px; border-left: 1px solid {BORDER}; }}
.f1-kpi:first-child {{ border-left: none; }}
.f1-kpi .l {{ font-size: 13px; color: {MUTED}; font-weight: 500; }}
.f1-kpi .v {{ font-size: 26px; color: {INK}; font-weight: 400; font-variant-numeric: tabular-nums; line-height: 1.25;
  margin-top: 4px; }}
.f1-kpi .v small {{ font-size: 14px; color: {MUTED}; margin-left: 3px; }}
.f1-kpi .s {{ font-size: 13px; color: {MUTED}; margin-top: 2px; }}
.f1-kpi .s.pos {{ color: {POSITIVE}; }} .f1-kpi .s.neg {{ color: {NEGATIVE}; }}
.f1-brand {{ display: flex; align-items: center; gap: 10px; padding: 4px 0 14px; }}
.f1-brand b {{ font-size: 15px; font-weight: 600; color: {INK}; }}
.f1-brand span {{ display: block; font-size: 12px; color: {MUTED}; }}
.f1-note, [data-testid="stSidebar"] .f1-note {{ font-size: 13px !important; color: {MUTED}; margin-top: 4px;
  line-height: 1.45; }}
.st-key-f1search {{ position: fixed; top: 8px; left: 50%; transform: translateX(-50%); width: min(460px, 46vw) !important;
  z-index: 1000001; }}
.st-key-f1search [data-baseweb="select"] > div {{ min-height: 36px; border-radius: 999px; background: #1a1b1e; }}
.st-key-f1help {{ position: fixed; top: 10px; right: 56px; z-index: 1000002; width: auto !important; }}
.st-key-f1help button {{ width: 34px; height: 34px; min-height: 34px; padding: 0; border-radius: 999px;
  font-weight: 600; font-size: 16px; }}
.st-key-f1help button [data-testid="stIconMaterial"] {{ display: none; }}
[data-testid="stPopoverBody"] {{ width: min(600px, 92vw); max-height: 75vh; overflow-y: auto; }}
@media (max-width: 640px) {{
  .st-key-f1help {{ right: 48px; }}
  .st-key-f1search {{ left: 64px; transform: none; width: calc(100vw - 168px) !important; }}
  [data-testid="stMain"] {{ margin: 0; border-radius: 0; border: none; }}
  .f1-kpis {{ grid-template-columns: 1fr 1fr !important; }}
  .f1-kpi:nth-child(odd) {{ border-left: none; }}
  .f1-kpi:nth-child(n+3) {{ border-top: 1px solid {BORDER}; }}
}}
</style>
"""

# the mark: a chequered timing line, two rows of squares in the accent and ink
MARK = f"""<svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
<rect width="28" height="28" rx="6" fill="{ACCENT}"/>
<g fill="{FRAME}"><rect x="6" y="8" width="4" height="4"/><rect x="14" y="8" width="4" height="4"/>
<rect x="10" y="12" width="4" height="4"/><rect x="18" y="12" width="4" height="4"/>
<rect x="6" y="16" width="4" height="4"/><rect x="14" y="16" width="4" height="4"/></g></svg>"""


def setup():
    st.markdown(CSS, unsafe_allow_html=True)
    alt.themes.register("f1", _altair_theme)
    alt.themes.enable("f1")


def brand(scope: str):
    from pathlib import Path
    st.logo(str(Path(__file__).with_name("logo.svg")), size="large")
    st.sidebar.markdown(f'<p class="f1-note">{html.escape(scope)}</p>', unsafe_allow_html=True)


def title(text: str, finding: str | None = None, scope: str | None = None):
    if scope:
        st.markdown(f'<p class="f1-scope">{html.escape(scope)}</p>', unsafe_allow_html=True)
    st.title(text)
    if finding:
        st.markdown(f'<p class="f1-finding">{finding}</p>', unsafe_allow_html=True)


def kpis(items: list[tuple]):
    """items: (label, value, unit=None, sub=None, tone=None) with tone 'pos' | 'neg' | None."""
    cells = []
    for it in items:
        label, value, unit, sub, tone = (list(it) + [None] * 5)[:5]
        u = f"<small>{html.escape(str(unit))}</small>" if unit else ""
        s = f'<div class="s {tone or ""}">{html.escape(str(sub))}</div>' if sub else ""
        cells.append(f'<div class="f1-kpi"><div class="l">{html.escape(str(label))}</div>'
                     f'<div class="v">{html.escape(str(value))}{u}</div>{s}</div>')
    st.markdown(f'<div class="f1-kpis" style="grid-template-columns: repeat({len(items)}, 1fr)">{"".join(cells)}</div>',
                unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<p class="f1-note">{text}</p>', unsafe_allow_html=True)


def dash(x, fmt="{:.2f}", unit=""):
    """Missing values print as a dash, never as 0 or nan."""
    try:
        if x is None or x != x:
            return "–"
    except TypeError:
        return "–"
    return fmt.format(x) + unit


def _altair_theme():
    return {"config": {
        "font": FONT,
        "background": SHEET,
        "view": {"stroke": None},
        "axis": {"labelColor": MUTED, "labelFontSize": 12, "titleColor": MUTED, "titleFontSize": 12,
                 "titleFontWeight": 500, "gridColor": RULE, "domain": False, "tickColor": RULE, "labelPadding": 6,
                 "tickCount": 6},
        "axisX": {"grid": False, "domain": True, "domainColor": "#4a4d53", "labelAngle": 0, "labelOverlap": "greedy"},
        "axisBand": {"grid": False},
        "legend": {"labelColor": INK, "labelFontSize": 12, "titleColor": MUTED, "titleFontSize": 12,
                   "titleFontWeight": 500, "symbolType": "square", "orient": "bottom", "direction": "horizontal"},
        "header": {"labelColor": INK, "labelFontSize": 13, "titleColor": MUTED},
        "title": {"color": INK, "fontSize": 14, "fontWeight": 600, "anchor": "start"},
        "bar": {"color": ACCENT, "cornerRadius": 0},
        "line": {"color": ACCENT, "strokeWidth": 2},
        "point": {"color": ACCENT, "filled": True, "size": 40},
        "circle": {"color": ACCENT},
        "text": {"color": INK, "fontSize": 11},
        "range": {"category": ["#4fb3c8", "#6f9fe0", "#d9a441", "#a3a6ad", "#4cc38a", "#f06272", "#8b9bb0", "#5fbf9f"]},
    }}


def help_button(page_key: str):
    """The "?" in the top-right corner: what this page answers, how to read it, every page, and a glossary."""
    from guide import GLOSSARY, PAGES
    info = PAGES.get(page_key, {})
    with st.container(key="f1help"):
        with st.popover(" ?", help="What is on this page and what the terms mean"):
            t1, t2, t3 = st.tabs(["This page", "All pages", "Glossary"])
            with t1:
                if info:
                    st.markdown(f"**{info['title']}**: {info['answers']}")
                    st.markdown("\n".join(f"- {line}" for line in info["read"]))
                else:
                    st.markdown("No guide for this page yet.")
            with t2:
                st.markdown("\n".join(f"- **{v['title']}**: {v['answers']}" for v in PAGES.values()))
            with t3:
                st.markdown("\n".join(f"- **{k}**: {v}" for k, v in GLOSSARY))
