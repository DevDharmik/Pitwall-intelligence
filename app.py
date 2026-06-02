"""
PitWall Intelligence — Formula 1 Brand Value Index
==================================================
M.Sc. capstone dashboard (Sprint 4 deliverable):
    "Predictive Modeling of Formula 1 Constructor Performance and
     Sponsorship Value Using Machine Learning."

Everything on screen is computed from the project's *real* artefacts —
no synthetic data:
    data/exports/bvi_scores.csv          Sprint 3 Brand Value Index (per team / season)
    data/exports/team_season_stats.csv   Sprint 1/2 enriched stats (wins, DNFs, qual gap…)
    reports/sprint2_all_models_metrics.csv   validated CV / test metrics
    reports/sprint2_gbr_test_preds.csv       Gradient-Boosting 2024 hold-out predictions
    models/gbr_total_points_v1.joblib        the fitted model → live feature importances
    reports/sprint3_*shap*.png               Sprint 3 SHAP beeswarm plots

Scope (authoritative):
    • Single data source : Jolpica-F1 API  ->  SQLite (pitwall.db)
    • Era               : V6 hybrid, 2014–2024
    • BVI, per constructor / season, min-max normalised:
          Performance (60%) = predicted points + podium probability
          Consistency (40%) = reliability + qualifying-to-race delta
    • Models            : Gradient Boosting (points), Logistic Reg. + Platt (podium)
    • Explainability    : SHAP attribution on both models

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================================================
#  Paths — resolve relative to this file so the app runs from anywhere
# ==========================================================================
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data" / "exports"
REPORTS_DIR = APP_DIR / "reports"
MODELS_DIR = APP_DIR / "models"


def _first_existing(*candidates: Path) -> Path | None:
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


def _resolve(env_var: str, *names: str, folder: Path = DATA_DIR) -> Path | None:
    """Locate a data file: env override -> folder -> app dir -> cwd."""
    env = os.environ.get(env_var)
    cands: list[Path] = [Path(env)] if env else []
    for n in names:
        cands += [folder / n, APP_DIR / n, Path.cwd() / n, Path.cwd() / "data" / "exports" / n]
    return _first_existing(*cands)


# ==========================================================================
#  Page config + brand palette
# ==========================================================================
st.set_page_config(
    page_title="PitWall Intelligence — F1 Brand Value Index",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

F1_RED = "#E10600"
CARBON = "#0B0B10"
INK_2 = "#15151E"
TXT = "#F2F2F5"
MUTED = "#8A8A98"

# Recognisable constructor livery colours (V6-hybrid era).
TEAM_COLORS: dict[str, str] = {
    "Mercedes": "#00D2BE",
    "Red Bull": "#1E5BC6",
    "Ferrari": "#DC0000",
    "McLaren": "#FF8700",
    "Williams": "#00A3E0",
    "Force India": "#FF80C7",
    "Racing Point": "#F596C8",
    "Aston Martin": "#229971",
    "Alpine F1 Team": "#0093CC",
    "Renault": "#FFD800",
    "Lotus F1": "#FFB800",
    "Alfa Romeo": "#B12039",
    "AlphaTauri": "#5E779E",
    "Toro Rosso": "#469BFF",
    "Sauber": "#52A0FF",
    "Haas F1 Team": "#B6BABD",
    "RB F1 Team": "#6692FF",
    "Caterham": "#0B6E4F",
    "Manor Marussia": "#C8102E",
    "Marussia": "#C8102E",
}
DEFAULT_TEAM_COLOR = "#9AA0AA"


def team_color(name: str) -> str:
    return TEAM_COLORS.get(name, DEFAULT_TEAM_COLOR)


# ==========================================================================
#  CSS — dark "pit-wall telemetry" aesthetic
#  (plain string, literal colours, so there are no f-string brace clashes)
# ==========================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Barlow:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root{
  --red:#E10600; --carbon:#0B0B10; --ink2:#15151E; --txt:#F2F2F5; --muted:#8A8A98;
}

/* ---- canvas: carbon + faint telemetry grid + corner glow ---- */
.stApp{
  background:
    radial-gradient(1200px 600px at 12% -8%, rgba(225,6,0,0.14), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(0,210,190,0.07), transparent 55%),
    linear-gradient(180deg, #0C0C12 0%, #08080C 100%);
  background-attachment: fixed;
}
[data-testid="stAppViewContainer"]>.main::before{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:
    linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
  background-size: 46px 46px;
  -webkit-mask-image: radial-gradient(ellipse at 50% 0%, #000 30%, transparent 80%);
  mask-image: radial-gradient(ellipse at 50% 0%, #000 30%, transparent 80%);
}
.block-container{ padding-top:1.4rem; padding-bottom:3rem; max-width:1280px; position:relative; z-index:1; }

/* ---- typography ---- */
html, body, [class*="css"]{ font-family:'Barlow', sans-serif; color:var(--txt); }
h1,h2,h3,h4{ font-family:'Rajdhani', sans-serif; letter-spacing:.01em; color:#fff; }
.stApp, p, span, label, li{ color:var(--txt); }

/* ---- hero ---- */
.pw-hero{ margin:.2rem 0 1.3rem; }
.pw-kicker{
  font-family:'JetBrains Mono', monospace; font-size:.74rem; letter-spacing:.34em;
  color:var(--red); text-transform:uppercase; display:flex; align-items:center; gap:.6rem;
}
.pw-kicker::before{ content:""; width:30px; height:2px; background:var(--red); display:inline-block; }
.pw-title{
  font-family:'Rajdhani', sans-serif; font-weight:700; line-height:.96;
  font-size:clamp(2.5rem, 6vw, 4.6rem); margin:.15rem 0 .1rem; color:#fff;
  text-transform:uppercase;
}
.pw-title .lo{ color:var(--red); }
.pw-sub{ color:var(--muted); font-size:clamp(.92rem,1.6vw,1.06rem); max-width:760px; font-weight:300; }
.pw-accent{
  height:4px; width:100%; margin-top:1rem; border-radius:3px;
  background:linear-gradient(90deg, var(--red) 0%, var(--red) 28%, #00D2BE 28%, #00D2BE 46%,
            #FF8700 46%, #FF8700 60%, #1E5BC6 60%, #1E5BC6 74%, rgba(255,255,255,.12) 74%);
  background-size:200% 100%; animation:pw-slide 9s linear infinite; opacity:.85;
}
@keyframes pw-slide{ 0%{background-position:0% 0} 100%{background-position:-200% 0} }

/* ---- KPI strip ---- */
.pw-kpis{ display:flex; flex-wrap:wrap; gap:14px; margin:.2rem 0 1.4rem; }
.pw-kpi{
  flex:1 1 200px; min-width:180px; background:linear-gradient(160deg, rgba(255,255,255,.045), rgba(255,255,255,.012));
  border:1px solid rgba(255,255,255,.08); border-left:3px solid var(--red);
  border-radius:10px; padding:14px 16px 13px; position:relative; overflow:hidden;
  opacity:0; transform:translateY(10px); animation:pw-rise .55s ease forwards;
}
.pw-kpi::after{ content:""; position:absolute; right:-30px; top:-30px; width:90px; height:90px;
  background:radial-gradient(circle, rgba(255,255,255,.06), transparent 70%); }
.pw-kpi .lab{ font-family:'JetBrains Mono', monospace; font-size:.62rem; letter-spacing:.18em;
  text-transform:uppercase; color:var(--muted); }
.pw-kpi .val{ font-family:'Rajdhani', sans-serif; font-weight:700; font-size:1.78rem; line-height:1.05;
  margin-top:.18rem; color:#fff; }
.pw-kpi .sub{ font-family:'JetBrains Mono', monospace; font-size:.72rem; color:var(--muted); margin-top:.18rem; }
@keyframes pw-rise{ to{ opacity:1; transform:translateY(0); } }

/* ---- timing tower ---- */
.pw-tower{ display:flex; flex-direction:column; gap:8px; margin-top:.4rem; }
.pw-row{
  display:grid; grid-template-columns:42px 12px 1fr 86px; align-items:center; gap:12px;
  background:linear-gradient(90deg, rgba(255,255,255,.04), rgba(255,255,255,.015));
  border:1px solid rgba(255,255,255,.07); border-radius:9px; padding:9px 14px 9px 8px;
  opacity:0; transform:translateX(-8px); animation:pw-in .5s ease forwards;
  transition:border-color .18s ease, background .18s ease;
}
.pw-row:hover{ border-color:rgba(255,255,255,.22); background:linear-gradient(90deg, rgba(255,255,255,.07), rgba(255,255,255,.02)); }
.pw-pos{ font-family:'JetBrains Mono', monospace; font-weight:700; font-size:1.15rem; text-align:center; color:#fff; }
.pw-pos small{ display:block; font-size:.5rem; color:var(--muted); letter-spacing:.12em; font-weight:400; }
.pw-stripe{ width:6px; height:34px; border-radius:3px; }
.pw-mid .nm{ font-family:'Rajdhani', sans-serif; font-weight:600; font-size:1.12rem; color:#fff; line-height:1; }
.pw-track{ position:relative; height:9px; border-radius:5px; background:rgba(255,255,255,.07); margin-top:7px; overflow:hidden; }
.pw-fill{ position:absolute; left:0; top:0; height:100%; border-radius:5px; width:0;
  animation:pw-grow 1.1s cubic-bezier(.2,.8,.2,1) forwards; }
.pw-split{ font-family:'JetBrains Mono', monospace; font-size:.62rem; color:var(--muted); margin-top:5px; letter-spacing:.04em; }
.pw-bvi{ text-align:right; }
.pw-bvi b{ font-family:'JetBrains Mono', monospace; font-size:1.32rem; font-weight:700; color:#fff; }
.pw-bvi span{ display:block; font-family:'JetBrains Mono', monospace; font-size:.56rem; color:var(--muted); letter-spacing:.16em; }
@keyframes pw-in{ to{ opacity:1; transform:translateX(0); } }
@keyframes pw-grow{ to{ width:var(--w); } }

/* ---- chips / tags ---- */
.pw-tag{ display:inline-block; background:var(--red); color:#fff; font-family:'JetBrains Mono', monospace;
  font-weight:500; padding:3px 9px; border-radius:4px; font-size:.62rem; letter-spacing:.12em; }
.pw-chip{ display:inline-block; border:1px solid rgba(255,255,255,.16); border-radius:20px;
  padding:3px 11px; font-size:.72rem; color:var(--muted); margin:2px 4px 2px 0; font-family:'JetBrains Mono', monospace; }
.pw-note{ color:var(--muted); font-size:.9rem; font-weight:300; }

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"]{ gap:6px; border-bottom:1px solid rgba(255,255,255,.08); }
.stTabs [data-baseweb="tab"]{
  background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); border-bottom:none;
  border-radius:8px 8px 0 0; padding:8px 16px; font-family:'Rajdhani', sans-serif; font-weight:600;
  font-size:1rem; color:var(--muted); letter-spacing:.02em;
}
.stTabs [aria-selected="true"]{ background:rgba(225,6,0,.14); color:#fff; border-color:rgba(225,6,0,.5); }

/* ---- sidebar ---- */
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0E0E15,#0A0A0F); border-right:1px solid rgba(255,255,255,.07); }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3{ color:#fff; }
[data-testid="stMetricValue"]{ font-family:'Rajdhani', sans-serif; font-weight:700; }

hr{ border-color:rgba(255,255,255,.08); }

/* ---- responsive ---- */
@media (max-width:680px){
  .pw-kpi{ flex:1 1 100%; }
  .pw-row{ grid-template-columns:34px 8px 1fr 66px; gap:8px; padding:8px; }
  .block-container{ padding-left:.6rem; padding-right:.6rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================================================
#  Data loading  (cached) — REAL artefacts only
# ==========================================================================
@st.cache_data(show_spinner=False)
def load_bvi() -> pd.DataFrame | None:
    p = _resolve("BVI_CSV", "bvi_scores.csv")
    if not p:
        return None
    df = pd.read_csv(p)
    df["season"] = df["season"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_team_stats() -> pd.DataFrame | None:
    p = _resolve("TEAM_STATS_CSV", "team_season_stats.csv")
    if not p:
        return None
    df = pd.read_csv(p)
    df["season"] = df["season"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_merged() -> pd.DataFrame | None:
    bvi, stats = load_bvi(), load_team_stats()
    if bvi is None:
        return None
    if stats is None:
        return bvi.copy()
    extra = [c for c in stats.columns if c not in bvi.columns or c in ("season", "name")]
    merged = bvi.merge(stats[extra], on=["season", "name"], how="left")
    return merged


@st.cache_data(show_spinner=False)
def load_model_metrics() -> pd.DataFrame | None:
    p = _resolve("METRICS_CSV", "sprint2_all_models_metrics.csv", folder=REPORTS_DIR)
    if not p:
        p = _first_existing(REPORTS_DIR / "sprint2_all_models_metrics.csv")
    if not p:
        return None
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_gbr_preds() -> pd.DataFrame | None:
    p = _first_existing(REPORTS_DIR / "sprint2_gbr_test_preds.csv", APP_DIR / "sprint2_gbr_test_preds.csv")
    return pd.read_csv(p) if p else None


@st.cache_resource(show_spinner=False)
def load_gbr_importances() -> tuple[pd.DataFrame | None, int | None]:
    """Pull genuine feature importances from the fitted GBR bundle."""
    p = _first_existing(MODELS_DIR / "gbr_total_points_v1.joblib")
    if not p:
        return None, None
    try:
        import joblib

        bundle = joblib.load(p)
        model = bundle.get("model")
        feats = bundle.get("features")
        test_season = bundle.get("test_season")
        if model is None or not hasattr(model, "feature_importances_"):
            return None, test_season
        imp = pd.DataFrame(
            {"feature": feats, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=True)
        return imp, test_season
    except Exception:
        return None, None


def find_report_fig(filename: str) -> Path | None:
    return _first_existing(REPORTS_DIR / filename, APP_DIR / filename, Path.cwd() / "reports" / filename)


# Plotly dark template tuned to the brand -----------------------------------
def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Barlow, sans-serif", color=TXT, size=13),
        margin=dict(l=10, r=10, t=44, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        title=dict(font=dict(family="Rajdhani, sans-serif", size=18, color="#fff")),
        hoverlabel=dict(font=dict(family="JetBrains Mono, monospace", size=12)),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


# ==========================================================================
#  Load everything once
# ==========================================================================
df = load_merged()
metrics = load_model_metrics()
gbr_preds = load_gbr_preds()
imp_df, model_test_season = load_gbr_importances()

# ----- hard stop if the real data is missing (no synthetic fabrication) -----
if df is None:
    st.error(
        "Could not find **`data/exports/bvi_scores.csv`**.\n\n"
        "Run this app from the repository root, or set the `BVI_CSV` "
        "environment variable to point at the Sprint-3 export."
    )
    st.stop()

SEASONS = sorted(df["season"].unique())
HAS_STATS = "total_wins" in df.columns


# ==========================================================================
#  Sidebar — controls
# ==========================================================================
with st.sidebar:
    st.markdown("### 🏁 PIT WALL")
    st.markdown(
        f"<span class='pw-chip'>Jolpica-F1 API</span>"
        f"<span class='pw-chip'>{SEASONS[0]}–{SEASONS[-1]}</span>"
        f"<span class='pw-chip'>{df.shape[0]} team-seasons</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    view_all = st.toggle("All-era view", value=False, help="Aggregate every season instead of one")
    if view_all:
        season = None
        st.caption(f"Showing the full {SEASONS[0]}–{SEASONS[-1]} hybrid era.")
    else:
        season = st.select_slider("Season", options=SEASONS, value=SEASONS[-1])

    st.markdown("---")
    st.markdown("##### Brand Value Index")
    st.markdown(
        "<span class='pw-note'>BVI = <b>Performance 60%</b> (predicted points + "
        "podium probability) + <b>Consistency 40%</b> (reliability + qualifying-to-race "
        "delta), min-max normalised per season and scaled to 0–100.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<span class='pw-note'>M.Sc. Data Science capstone · "
        "University of Europe for Applied Sciences.<br>"
        "<a href='https://github.com/DevDharmik/Pitwall-intelligence' "
        "style='color:#E10600;text-decoration:none;'>github.com/DevDharmik/Pitwall-intelligence ↗</a>"
        "</span>",
        unsafe_allow_html=True,
    )


# ==========================================================================
#  Hero
# ==========================================================================
ctx = "ALL ERA · 2014–2024" if view_all else f"SEASON {season}"
st.markdown(
    f"""
<div class="pw-hero">
  <div class="pw-kicker">PitWall Intelligence · {ctx}</div>
  <div class="pw-title">F1 BRAND <span class="lo">VALUE</span> INDEX</div>
  <div class="pw-sub">A sponsor-facing read on which Formula&nbsp;1 constructors convert speed and
  reliability into measurable value — powered by machine-learning predictions and SHAP explainability
  across the V6-hybrid era.</div>
  <div class="pw-accent"></div>
</div>
""",
    unsafe_allow_html=True,
)


# ==========================================================================
#  Helpers for the season slice
# ==========================================================================
def season_slice() -> pd.DataFrame:
    if view_all:
        agg = {
            "BVI_100": "mean",
            "performance_score": "mean",
            "consistency_score": "mean",
            "total_points": "sum",
        }
        if HAS_STATS:
            agg.update(
                {
                    "total_wins": "sum",
                    "total_podiums": "sum",
                    "total_dnfs": "sum",
                    "finish_rate": "mean",
                    "avg_gap_to_pole": "mean",
                    "q3_appearances": "sum",
                    "races_entered": "sum",
                }
            )
        g = (
            df.groupby("name")
            .agg({k: v for k, v in agg.items() if k in df.columns})
            .reset_index()
            .sort_values("BVI_100", ascending=False)
        )
        g["bvi_rank"] = range(1, len(g) + 1)
        return g
    return df[df["season"] == season].sort_values("BVI_100", ascending=False).reset_index(drop=True)


cur = season_slice()


def kpi(label: str, value: str, sub: str = "", delay: float = 0.0, accent: str = F1_RED) -> str:
    return (
        f"<div class='pw-kpi' style='animation-delay:{delay}s;border-left-color:{accent};'>"
        f"<div class='lab'>{label}</div><div class='val'>{value}</div>"
        f"<div class='sub'>{sub}</div></div>"
    )


# ==========================================================================
#  KPI strip — leaders for the current slice
# ==========================================================================
top = cur.iloc[0]
champ_color = team_color(top["name"])

# biggest overperformer: best (champ_rank - bvi_rank) when we have champ_rank
over_name = "—"
over_txt = "—"
if not view_all and "champ_rank" in cur.columns:
    tmp = cur.copy()
    tmp["delta"] = tmp["champ_rank"] - tmp["bvi_rank"]
    o = tmp.sort_values("delta", ascending=False).iloc[0]
    if o["delta"] > 0:
        over_txt = f"P{int(o['champ_rank'])} → BVI P{int(o['bvi_rank'])}"
        over_name = o["name"]
    else:
        over_name = "No upset"
        over_txt = "ranks aligned"

most_consistent = cur.sort_values("consistency_score", ascending=False).iloc[0]

kpis = [
    kpi("Top brand value", top["name"], f"BVI {top['BVI_100']:.1f} / 100", 0.00, champ_color),
    kpi(
        "Most consistent",
        most_consistent["name"],
        f"consistency {most_consistent['consistency_score']:.2f}",
        0.08,
        team_color(most_consistent["name"]),
    ),
]
if not view_all and over_name not in ("—", "No upset"):
    kpis.append(kpi("Biggest overperformer", over_name, over_txt, 0.16, team_color(over_name)))
if HAS_STATS:
    wins_total = int(cur["total_wins"].sum()) if "total_wins" in cur else 0
    kpis.append(kpi("Wins on the grid", f"{wins_total}", "race victories", 0.24, "#00D2BE"))

st.markdown(f"<div class='pw-kpis'>{''.join(kpis)}</div>", unsafe_allow_html=True)


# ==========================================================================
#  Tabs
# ==========================================================================
tab_grid, tab_h2h, tab_quad, tab_evo, tab_model = st.tabs(
    ["  GRID  ", "  HEAD-TO-HEAD  ", "  PERF × CONSISTENCY  ", "  EVOLUTION  ", "  THE MODEL  "]
)

# --------------------------------------------------------------------------
# 1) GRID — the signature broadcast timing tower
# --------------------------------------------------------------------------
with tab_grid:
    left, right = st.columns([1.18, 1], gap="large")

    with left:
        st.markdown(
            f"#### Timing tower · <span class='pw-tag'>{'ALL ERA' if view_all else season}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<span class='pw-note'>Ranked by Brand Value Index. The bar is BVI/100; "
            "the split below it shows the Performance vs Consistency contribution.</span>",
            unsafe_allow_html=True,
        )
        bmax = max(float(cur["BVI_100"].max()), 1.0)
        rows = []
        for i, r in cur.iterrows():
            c = team_color(r["name"])
            pos = i + 1
            w = max(4.0, r["BVI_100"] / bmax * 100.0)
            perf = r.get("performance_score", float("nan"))
            cons = r.get("consistency_score", float("nan"))
            split = (
                f"PERF {perf:.2f} · CONS {cons:.2f}"
                if pd.notna(perf) and pd.notna(cons)
                else ""
            )
            delay = min(i * 0.05, 0.8)
            rows.append(
                f"<div class='pw-row' style='animation-delay:{delay}s'>"
                f"<div class='pw-pos'>{pos}<small>BVI</small></div>"
                f"<div class='pw-stripe' style='background:{c}'></div>"
                f"<div class='pw-mid'><div class='nm'>{r['name']}</div>"
                f"<div class='pw-track'><div class='pw-fill' style='--w:{w:.1f}%;"
                f"background:linear-gradient(90deg,{c},{c}cc)'></div></div>"
                f"<div class='pw-split'>{split}</div></div>"
                f"<div class='pw-bvi'><b>{r['BVI_100']:.1f}</b><span>INDEX</span></div>"
                f"</div>"
            )
        st.markdown(f"<div class='pw-tower'>{''.join(rows)}</div>", unsafe_allow_html=True)

    with right:
        st.markdown("#### Value vs. the championship")
        st.markdown(
            "<span class='pw-note'>Does brand value track the actual standings? Points "
            "lying off the diagonal are teams the BVI rates differently than the title fight did.</span>",
            unsafe_allow_html=True,
        )
        if not view_all and "champ_rank" in cur.columns:
            d = cur.copy()
            fig = go.Figure()
            lim = [0.5, len(d) + 0.5]
            fig.add_trace(
                go.Scatter(
                    x=lim, y=lim, mode="lines",
                    line=dict(color="rgba(255,255,255,0.25)", dash="dot"),
                    hoverinfo="skip", showlegend=False,
                )
            )
            for _, r in d.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[r["champ_rank"]], y=[r["bvi_rank"]], mode="markers+text",
                        marker=dict(size=15, color=team_color(r["name"]),
                                    line=dict(color="white", width=1)),
                        text=[r["name"][:3].upper()], textposition="top center",
                        textfont=dict(family="JetBrains Mono, monospace", size=9, color=MUTED),
                        name=r["name"],
                        hovertemplate=f"<b>{r['name']}</b><br>Championship P%{{x}}<br>BVI P%{{y}}<extra></extra>",
                        showlegend=False,
                    )
                )
            fig.update_xaxes(title="Championship rank", autorange="reversed", dtick=1)
            fig.update_yaxes(title="BVI rank", autorange="reversed", dtick=1)
            st.plotly_chart(style_fig(fig, 470), use_container_width=True)
            st.caption("Above the line → BVI ranks the team higher than the championship did.")
        else:
            d = cur.head(12).sort_values("BVI_100")
            fig = go.Figure(
                go.Bar(
                    x=d["BVI_100"], y=d["name"], orientation="h",
                    marker=dict(color=[team_color(n) for n in d["name"]]),
                    text=[f"{v:.1f}" for v in d["BVI_100"]],
                    textposition="outside",
                    textfont=dict(family="JetBrains Mono, monospace", color="#fff"),
                    hovertemplate="<b>%{y}</b><br>mean BVI %{x:.1f}<extra></extra>",
                )
            )
            fig.update_layout(title="Mean BVI · full hybrid era")
            st.plotly_chart(style_fig(fig, 470), use_container_width=True)

# --------------------------------------------------------------------------
# 2) HEAD-TO-HEAD — radar + stat duel
# --------------------------------------------------------------------------
with tab_h2h:
    st.markdown("#### Constructor duel")
    st.markdown(
        "<span class='pw-note'>Pick two constructors and compare them across the dimensions that "
        "feed the index. Radar axes are normalised 0–1 within the current slice "
        "(qualifying gap is inverted so further-out is always better).</span>",
        unsafe_allow_html=True,
    )

    names = list(cur["name"])
    c1, c2 = st.columns(2)
    a = c1.selectbox("Constructor A", names, index=0)
    b = c2.selectbox("Constructor B", names, index=min(1, len(names) - 1))

    radar_specs = [
        ("BVI_100", "Brand value", False),
        ("performance_score", "Performance", False),
        ("consistency_score", "Consistency", False),
    ]
    if HAS_STATS:
        radar_specs += [
            ("total_wins", "Wins", False),
            ("total_podiums", "Podiums", False),
            ("finish_rate", "Reliability", False),
            ("avg_gap_to_pole", "Qualifying pace", True),
        ]
    radar_specs = [(c, lbl, inv) for c, lbl, inv in radar_specs if c in cur.columns]

    def norm_series(s: pd.Series, invert: bool) -> pd.Series:
        lo, hi = s.min(), s.max()
        if hi - lo < 1e-9:
            base = pd.Series(0.5, index=s.index)
        else:
            base = (s - lo) / (hi - lo)
        return 1 - base if invert else base

    norm = cur.copy()
    for col, _, inv in radar_specs:
        norm[col + "__n"] = norm_series(norm[col], inv)

    ra = norm[norm["name"] == a].iloc[0]
    rb = norm[norm["name"] == b].iloc[0]
    cats = [lbl for _, lbl, _ in radar_specs]
    va = [ra[c + "__n"] for c, _, _ in radar_specs]
    vb = [rb[c + "__n"] for c, _, _ in radar_specs]

    col_radar, col_table = st.columns([1.05, 1], gap="large")

    with col_radar:
        fig = go.Figure()
        for nm, vals, clr in [(a, va, team_color(a)), (b, vb, team_color(b))]:
            fig.add_trace(
                go.Scatterpolar(
                    r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
                    name=nm, line=dict(color=clr, width=2), opacity=0.55,
                )
            )
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(255,255,255,0.02)",
                radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                                gridcolor="rgba(255,255,255,0.12)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.12)",
                                 tickfont=dict(family="Rajdhani, sans-serif", size=13, color="#fff")),
            ),
            showlegend=True,
        )
        st.plotly_chart(style_fig(fig, 440), use_container_width=True)

    with col_table:
        st.markdown("##### Stat by stat")
        duel_cols = [
            ("BVI_100", "BVI / 100", "{:.1f}", "high"),
            ("performance_score", "Performance", "{:.2f}", "high"),
            ("consistency_score", "Consistency", "{:.2f}", "high"),
        ]
        if HAS_STATS:
            duel_cols += [
                ("total_points", "Points", "{:.0f}", "high"),
                ("total_wins", "Wins", "{:.0f}", "high"),
                ("total_podiums", "Podiums", "{:.0f}", "high"),
                ("total_dnfs", "DNFs", "{:.0f}", "low"),
                ("finish_rate", "Finish rate", "{:.0%}", "high"),
                ("avg_gap_to_pole", "Avg gap to pole (s)", "{:.2f}", "low"),
            ]
        ra2 = cur[cur["name"] == a].iloc[0]
        rb2 = cur[cur["name"] == b].iloc[0]
        rowsH = [
            f"<tr><th style='text-align:left;color:{MUTED};font-family:JetBrains Mono;"
            f"font-weight:500;font-size:.7rem;padding:6px 8px'>METRIC</th>"
            f"<th style='color:{team_color(a)};font-family:Rajdhani;font-size:1rem;padding:6px 8px'>{a}</th>"
            f"<th style='color:{team_color(b)};font-family:Rajdhani;font-size:1rem;padding:6px 8px'>{b}</th></tr>"
        ]
        for col, lbl, fmt, better in duel_cols:
            if col not in cur.columns:
                continue
            xa, xb = ra2[col], rb2[col]
            if pd.isna(xa) or pd.isna(xb):
                continue
            a_win = (xa > xb) if better == "high" else (xa < xb)
            tie = abs(xa - xb) < 1e-9
            sa = "color:#fff;font-weight:700" if (a_win and not tie) else "color:#9AA0AA"
            sb = "color:#fff;font-weight:700" if (not a_win and not tie) else "color:#9AA0AA"
            rowsH.append(
                f"<tr style='border-top:1px solid rgba(255,255,255,.06)'>"
                f"<td style='text-align:left;color:{MUTED};font-size:.82rem;padding:6px 8px'>{lbl}</td>"
                f"<td style='font-family:JetBrains Mono;font-size:.92rem;padding:6px 8px;{sa}'>{fmt.format(xa)}</td>"
                f"<td style='font-family:JetBrains Mono;font-size:.92rem;padding:6px 8px;{sb}'>{fmt.format(xb)}</td></tr>"
            )
        st.markdown(
            "<table style='width:100%;border-collapse:collapse'>" + "".join(rowsH) + "</table>",
            unsafe_allow_html=True,
        )
        st.caption("Bold = the stronger constructor on that metric (DNFs & gap-to-pole: lower is better).")

# --------------------------------------------------------------------------
# 3) PERFORMANCE × CONSISTENCY quadrant
# --------------------------------------------------------------------------
with tab_quad:
    st.markdown("#### The two engines of brand value")
    st.markdown(
        "<span class='pw-note'>The index rests on two axes. Speed alone isn't enough — sponsors "
        "reward teams that are <i>both</i> fast and dependable. Bubble size = championship points; "
        "the dashed lines split the field at the slice's mid-points.</span>",
        unsafe_allow_html=True,
    )

    d = cur.copy()
    size_col = "total_points" if "total_points" in d.columns else None
    px_kw = dict(
        x="performance_score", y="consistency_score",
        color="name", text="name",
        color_discrete_map={n: team_color(n) for n in d["name"]},
    )
    if size_col:
        px_kw["size"] = d[size_col].clip(lower=1)
        px_kw["size_max"] = 34
    fig = px.scatter(d, **px_kw)
    fig.update_traces(
        textposition="top center",
        textfont=dict(family="JetBrains Mono, monospace", size=9, color=MUTED),
        marker=dict(line=dict(color="rgba(255,255,255,0.5)", width=1)),
    )
    mx, my = d["performance_score"].mean(), d["consistency_score"].mean()
    fig.add_vline(x=mx, line=dict(color="rgba(255,255,255,0.2)", dash="dash"))
    fig.add_hline(y=my, line=dict(color="rgba(255,255,255,0.2)", dash="dash"))
    ann = [
        (d["performance_score"].max(), d["consistency_score"].max(), "DOMINANT", "#00D2BE"),
        (d["performance_score"].min(), d["consistency_score"].max(), "RELIABLE, NOT FAST", MUTED),
        (d["performance_score"].max(), d["consistency_score"].min(), "FAST, FRAGILE", F1_RED),
        (d["performance_score"].min(), d["consistency_score"].min(), "BACKMARKERS", MUTED),
    ]
    for x, y, t, c in ann:
        fig.add_annotation(
            x=x, y=y, text=t, showarrow=False,
            font=dict(family="JetBrains Mono, monospace", size=10, color=c),
            opacity=0.8,
        )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Performance score  →")
    fig.update_yaxes(title="Consistency score  →")
    st.plotly_chart(style_fig(fig, 560), use_container_width=True)

# --------------------------------------------------------------------------
# 4) EVOLUTION — BVI trajectories
# --------------------------------------------------------------------------
with tab_evo:
    st.markdown("#### Eleven seasons of the hybrid era")
    st.markdown(
        "<span class='pw-note'>How brand value shifted from the Mercedes dynasty to Red Bull and "
        "on to McLaren's 2024 resurgence. Pick the constructors you want to trace.</span>",
        unsafe_allow_html=True,
    )

    persistent = df.groupby("name")["season"].nunique().sort_values(ascending=False)
    default_teams = [t for t in ["Mercedes", "Red Bull", "Ferrari", "McLaren"] if t in persistent.index]
    chosen = st.multiselect(
        "Constructors", options=list(persistent.index), default=default_teams or list(persistent.index[:4])
    )
    if chosen:
        d = df[df["name"].isin(chosen)].sort_values("season")
        fig = go.Figure()
        for nm in chosen:
            sub = d[d["name"] == nm]
            fig.add_trace(
                go.Scatter(
                    x=sub["season"], y=sub["BVI_100"], mode="lines+markers",
                    name=nm, line=dict(color=team_color(nm), width=3),
                    marker=dict(size=7, line=dict(color="#0B0B10", width=1)),
                    hovertemplate=f"<b>{nm}</b><br>%{{x}} · BVI %{{y:.1f}}<extra></extra>",
                )
            )
        fig.add_vrect(x0=2013.5, x1=2021.5, fillcolor="#00D2BE", opacity=0.05, line_width=0,
                      annotation_text="Mercedes titles", annotation_position="top left",
                      annotation_font=dict(family="JetBrains Mono", size=9, color="#00D2BE"))
        fig.add_vrect(x0=2021.5, x1=2023.5, fillcolor="#1E5BC6", opacity=0.06, line_width=0,
                      annotation_text="Red Bull titles", annotation_position="top left",
                      annotation_font=dict(family="JetBrains Mono", size=9, color="#6692FF"))
        fig.update_xaxes(title="Season", dtick=1)
        fig.update_yaxes(title="BVI / 100", range=[0, 105])
        st.plotly_chart(style_fig(fig, 520), use_container_width=True)
    else:
        st.info("Select at least one constructor to draw its trajectory.")

# --------------------------------------------------------------------------
# 5) THE MODEL — real metrics, predictions, importances, SHAP
# --------------------------------------------------------------------------
with tab_model:
    st.markdown("#### Under the hood")
    st.markdown(
        "<span class='pw-note'>The Performance half of the index is driven by a Gradient-Boosting "
        "regressor predicting season points, trained on 2014–2023 and validated on a "
        f"{model_test_season or 2024} hold-out. These are the project's actual results.</span>",
        unsafe_allow_html=True,
    )

    if metrics is not None:
        gbr = metrics[metrics["model"].str.contains("Gradient", case=False)]
        if not gbr.empty:
            g = gbr.iloc[0]
            mcards = [
                kpi("GBR · test R²", f"{g['test_r2']:.3f}", "variance explained (hold-out)", 0.0, "#00D2BE"),
                kpi("GBR · CV R²", f"{g['cv_r2']:.3f}", "5-fold cross-validation", 0.08, "#00D2BE"),
                kpi("GBR · test RMSE", f"{g['test_rmse']:.1f}", "points error", 0.16),
                kpi("GBR · test MAE", f"{g['test_mae']:.1f}", "points error", 0.24),
            ]
            st.markdown(f"<div class='pw-kpis'>{''.join(mcards)}</div>", unsafe_allow_html=True)

    cL, cR = st.columns(2, gap="large")

    with cL:
        st.markdown("##### Model leaderboard")
        if metrics is not None:
            m = metrics.sort_values("test_r2")
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    y=m["model"], x=m["test_r2"], orientation="h", name="Test R²",
                    marker=dict(color=[F1_RED if "Gradient" in x else "rgba(255,255,255,.35)" for x in m["model"]]),
                    text=[f"{v:.3f}" for v in m["test_r2"]], textposition="outside",
                    textfont=dict(family="JetBrains Mono, monospace", color="#fff"),
                    hovertemplate="<b>%{y}</b><br>test R² %{x:.3f}<extra></extra>",
                )
            )
            fig.update_xaxes(title="Test R²", range=[0, 1.08])
            fig.update_layout(title="Gradient Boosting wins")
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)
            show = metrics.copy()
            for c in ["cv_rmse", "cv_r2", "cv_mae", "test_rmse", "test_r2", "test_mae"]:
                if c in show.columns:
                    show[c] = show[c].round(3)
            st.dataframe(show, hide_index=True, use_container_width=True)
        else:
            st.info("`reports/sprint2_all_models_metrics.csv` not found.")

    with cR:
        st.markdown("##### What drives predicted points")
        if imp_df is not None:
            fig = go.Figure(
                go.Bar(
                    x=imp_df["importance"], y=imp_df["feature"], orientation="h",
                    marker=dict(
                        color=imp_df["importance"],
                        colorscale=[[0, "rgba(225,6,0,0.35)"], [1, F1_RED]],
                        line=dict(color="rgba(255,255,255,0.15)", width=1),
                    ),
                    text=[f"{v:.1%}" for v in imp_df["importance"]], textposition="outside",
                    textfont=dict(family="JetBrains Mono, monospace", color="#fff"),
                    hovertemplate="<b>%{y}</b><br>importance %{x:.1%}<extra></extra>",
                )
            )
            fig.update_xaxes(title="Gini importance", range=[0, max(imp_df["importance"]) * 1.25])
            fig.update_layout(title="GBR feature importance")
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)
            top_feat = imp_df.iloc[-1]
            st.markdown(
                f"<span class='pw-note'><b>{top_feat['feature']}</b> alone accounts for "
                f"{top_feat['importance']:.0%} of the model — grid position dominates, exactly what "
                f"the SHAP analysis confirms below.</span>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Feature importances are read from `models/gbr_total_points_v1.joblib`.")

    if gbr_preds is not None and not gbr_preds.empty:
        st.markdown("##### 2024 hold-out · predicted vs. actual points")
        d = gbr_preds.sort_values("total_points", ascending=True)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=d["name"], x=d["total_points"], orientation="h", name="Actual",
                marker=dict(color=[team_color(n) for n in d["name"]]),
                hovertemplate="<b>%{y}</b><br>actual %{x:.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                y=d["name"], x=d["gbr_pred"], mode="markers", name="GBR prediction",
                marker=dict(symbol="diamond", size=12, color="#fff",
                            line=dict(color="#0B0B10", width=1)),
                hovertemplate="<b>%{y}</b><br>predicted %{x:.0f}<extra></extra>",
            )
        )
        fig.update_xaxes(title="Season points")
        fig.update_layout(title="Bars = actual · diamonds = model", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(style_fig(fig, 430), use_container_width=True)

    st.markdown("##### SHAP explainability")
    st.markdown(
        "<span class='pw-note'>Beeswarm plots from Sprint&nbsp;3 — every dot is a team-season, "
        "coloured by feature value. Both the points regressor and the podium classifier are "
        "led by where a car starts: <b>starting grid position is the dominant feature</b>.</span>",
        unsafe_allow_html=True,
    )
    shap_figs = {
        "Points model — Gradient Boosting": "sprint3_01_shap_gbr_beeswarm.png",
        "Podium classifier — Logistic Regression": "sprint3_03_shap_podium_beeswarm.png",
    }
    scols = st.columns(2, gap="large")
    for col, (title, fname) in zip(scols, shap_figs.items()):
        with col:
            st.markdown(f"**{title}**")
            p = find_report_fig(fname)
            if p:
                st.image(str(p), use_container_width=True)
            else:
                st.info(f"`reports/{fname}` not found.")


# ==========================================================================
#  Footer
# ==========================================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div class='pw-note' style='display:flex;flex-wrap:wrap;gap:8px;justify-content:space-between;align-items:center'>"
    "<span><b style='color:#fff;font-family:Rajdhani'>PitWall Intelligence</b> · "
    "Predictive Modeling of F1 Constructor Performance &amp; Sponsorship Value</span>"
    "<span style='font-family:JetBrains Mono;font-size:.74rem'>"
    "Jolpica-F1 API · scikit-learn · SHAP · Streamlit · Plotly</span>"
    "</div>",
    unsafe_allow_html=True,
)
