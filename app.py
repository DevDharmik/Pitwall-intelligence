"""
PitWall Intelligence — Formula 1 Brand Value Index dashboard
============================================================
Sprint 4 deliverable. Wraps the Sprint 3 Brand Value Index (BVI) and the
SHAP explanations into an interactive, sponsor-facing tool.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Data:
    Loads the season-normalised BVI table produced in Sprint 3
    (notebook 07_bvi_shap.ipynb -> bvi_scores.csv). The app searches a few
    standard locations; point BVI_CSV at the file if it lives elsewhere.
"""

from pathlib import Path
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Config & theme
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="PitWall Intelligence — F1 Brand Value Index",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

F1_RED = "#E10600"
INK = "#15151E"
BVI_SCALE = "RdYlGn"  # red (low) -> green (high), matches the Sprint 3 heatmap

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1250px; }}
      h1, h2, h3 {{ color: {INK}; }}
      div[data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
      .pw-tag {{
          display:inline-block; background:{F1_RED}; color:white; font-weight:600;
          padding:2px 10px; border-radius:4px; font-size:0.72rem; letter-spacing:.04em;
      }}
      .pw-sub {{ color:#5b5b66; font-size:0.95rem; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
CSV_CANDIDATES = [
    os.environ.get("BVI_CSV", ""),
    "data/exports/bvi_scores.csv",
    "data/bvi_scores.csv",
    "bvi_scores.csv",
    str(Path(__file__).parent / "data" / "exports" / "bvi_scores.csv"),
]

REPORTS_DIRS = ["reports", "figures", str(Path(__file__).parent / "reports")]

SHAP_FIGURES = {
    "Points model — Gradient Boosting": "sprint3_01_shap_gbr_beeswarm.png",
    "Podium classifier — Logistic Regression": "sprint3_03_shap_podium_beeswarm.png",
}


@st.cache_data(show_spinner=False)
def load_bvi() -> pd.DataFrame:
    path = next((p for p in CSV_CANDIDATES if p and Path(p).exists()), None)
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["season"] = df["season"].astype(int)
    df["champ_rank"] = df["champ_rank"].astype(int)
    df["bvi_rank"] = df["bvi_rank"].astype(int)
    # express the two dimensions on the same 0-100 scale as the headline index
    df["performance_100"] = (df["performance_score"] * 100).round(1)
    df["consistency_100"] = (df["consistency_score"] * 100).round(1)
    # positive = BVI rates the team higher than the championship table did
    df["rank_delta"] = df["champ_rank"] - df["bvi_rank"]
    return df


def find_figure(filename: str):
    for d in REPORTS_DIRS:
        p = Path(d) / filename
        if p.exists():
            return str(p)
    return None


df = load_bvi()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<span class="pw-tag">PITWALL INTELLIGENCE</span>', unsafe_allow_html=True)
st.title("Formula 1 Brand Value Index")
st.markdown(
    '<p class="pw-sub">Turning on-track performance into an interpretable, comparable, '
    "sponsor-facing score — for a sponsorship market worth over <b>USD 1.8 billion</b> a year "
    "and an audience above <b>400 million</b>, still priced largely on perception.</p>",
    unsafe_allow_html=True,
)

if df.empty:
    st.error(
        "Could not find **bvi_scores.csv**. Place it at `data/exports/bvi_scores.csv` "
        "(or set the `BVI_CSV` environment variable to its path) and reload."
    )
    st.stop()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
seasons = sorted(df["season"].unique())
with st.sidebar:
    st.header("Controls")
    season = st.selectbox("Season", seasons, index=len(seasons) - 1)
    st.caption(
        "**Brand Value Index (BVI)**\n\n"
        "`0.60 ×` Performance `+ 0.40 ×` Consistency, min-max normalised "
        "**within each season** so a dominant era does not crush the midfield.\n\n"
        "**Performance** — predicted championship points + podium probability.\n\n"
        "**Consistency** — reliability + qualifying-to-race delta."
    )
    st.divider()
    st.caption(
        "Data: [Jolpica-F1 API](https://api.jolpi.ca/ergast/) · V6-hybrid era "
        f"{seasons[0]}–{seasons[-1]} · {len(df)} constructor-seasons."
    )

season_df = df[df["season"] == season].sort_values("bvi_rank").reset_index(drop=True)

# --------------------------------------------------------------------------
# KPI row
# --------------------------------------------------------------------------
leader = season_df.iloc[0]
riser = season_df.loc[season_df["rank_delta"].idxmax()]
faller = season_df.loc[season_df["rank_delta"].idxmin()]

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{season} BVI leader", leader["name"], f"{leader['BVI_100']:.1f} / 100")
c2.metric("Field size", f"{len(season_df)} teams", f"champion: {season_df.loc[season_df['champ_rank'].idxmin(),'name']}")
c3.metric(
    "Most underrated by points",
    riser["name"],
    f"+{int(riser['rank_delta'])} places vs P{int(riser['champ_rank'])}",
    help="Largest gap where the BVI ranks a team higher than the championship table — consistency the points miss.",
)
c4.metric(
    "Most flattered by points",
    faller["name"],
    f"{int(faller['rank_delta'])} places vs P{int(faller['champ_rank'])}",
    delta_color="inverse",
    help="Team the championship table ranks higher than the BVI does.",
)

st.divider()

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_stand, tab_map, tab_scatter, tab_traj, tab_shap = st.tabs(
    ["Season standings", "BVI map", "Performance vs Consistency",
     "Constructor trajectory", "What drives the score"]
)

# ---- Season standings ----------------------------------------------------
with tab_stand:
    st.subheader(f"{season} — Brand Value Index standings")
    plot_df = season_df.sort_values("BVI_100")
    fig = px.bar(
        plot_df, x="BVI_100", y="name", orientation="h",
        color="BVI_100", color_continuous_scale=BVI_SCALE, range_color=(0, 100),
        text="BVI_100", labels={"BVI_100": "BVI (0–100)", "name": ""},
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=460, coloraxis_showscale=False, margin=dict(l=10, r=30, t=10, b=10),
        xaxis_range=[0, 109], plot_bgcolor="white",
    )
    st.plotly_chart(fig, width='stretch')

    show = season_df[["bvi_rank", "name", "BVI_100", "performance_100",
                      "consistency_100", "champ_rank", "total_points"]].copy()
    show.columns = ["BVI rank", "Constructor", "BVI (0–100)", "Performance",
                    "Consistency", "Champ. rank", "Points"]
    st.dataframe(
        show, hide_index=True, width='stretch',
        column_config={
            "BVI (0–100)": st.column_config.ProgressColumn(
                "BVI (0–100)", min_value=0, max_value=100, format="%.1f"),
        },
    )

# ---- BVI map -------------------------------------------------------------
with tab_map:
    st.subheader(f"The BVI map — every constructor, {seasons[0]}–{seasons[-1]}")
    st.caption("Season-normalised on a 0–100 scale. Rows ordered by average BVI across the era.")
    pivot = df.pivot_table(index="name", columns="season", values="BVI_100", aggfunc="first")
    order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    heat = px.imshow(
        pivot, color_continuous_scale=BVI_SCALE, zmin=0, zmax=100,
        aspect="auto", text_auto=".0f",
        labels=dict(x="Season", y="Constructor", color="BVI"),
    )
    heat.update_xaxes(side="bottom", dtick=1, tickangle=0)
    heat.update_traces(textfont_size=10, xgap=1, ygap=1)
    heat.update_layout(height=720, margin=dict(l=10, r=10, t=10, b=10),
                       coloraxis_colorbar=dict(title="BVI"))
    st.plotly_chart(heat, width='stretch')

# ---- Performance vs Consistency -----------------------------------------
with tab_scatter:
    st.subheader(f"{season} — the two dimensions behind the score")
    st.caption(
        "Horizontal = Consistency (reliability + qualifying-to-race delta). "
        "Vertical = Performance (predicted points + podium probability). "
        "Bubble size and colour = the blended BVI."
    )
    med_p = season_df["performance_100"].median()
    med_c = season_df["consistency_100"].median()
    sc = px.scatter(
        season_df, x="consistency_100", y="performance_100",
        size="BVI_100", color="BVI_100", color_continuous_scale=BVI_SCALE,
        range_color=(0, 100), text="name", size_max=42,
        labels={"consistency_100": "Consistency (0–100)",
                "performance_100": "Performance (0–100)", "BVI_100": "BVI"},
        hover_data={"name": False, "BVI_100": ":.1f",
                    "champ_rank": True, "total_points": True},
    )
    sc.update_traces(textposition="top center", textfont_size=10)
    sc.add_hline(y=med_p, line_dash="dot", line_color="#bbb")
    sc.add_vline(x=med_c, line_dash="dot", line_color="#bbb")
    sc.update_layout(height=560, plot_bgcolor="white",
                     margin=dict(l=10, r=10, t=10, b=10),
                     xaxis_range=[-5, 105], yaxis_range=[-5, 105])
    st.plotly_chart(sc, width='stretch')
    st.caption(
        "Top-right: fast **and** reliable. Top-left: fast but fragile. "
        "Bottom-right: slow but dependable — where the Consistency dimension earns a team real BVI."
    )

# ---- Constructor trajectory ---------------------------------------------
with tab_traj:
    st.subheader("Constructor trajectory across the era")
    teams = sorted(df["name"].unique())
    default = ["Mercedes", "Red Bull", "Ferrari", "McLaren"]
    picks = st.multiselect(
        "Constructors", teams,
        default=[t for t in default if t in teams] or teams[:3],
    )
    if picks:
        tdf = df[df["name"].isin(picks)].sort_values(["name", "season"])
        line = px.line(
            tdf, x="season", y="BVI_100", color="name", markers=True,
            labels={"BVI_100": "BVI (0–100)", "season": "Season", "name": "Constructor"},
            hover_data={"champ_rank": True, "total_points": True, "season": False},
        )
        line.update_layout(height=480, plot_bgcolor="white",
                           margin=dict(l=10, r=10, t=10, b=10),
                           xaxis=dict(dtick=1), yaxis_range=[0, 105],
                           legend_title_text="")
        st.plotly_chart(line, width='stretch')
        st.caption("Hover any point for that season's championship finish and points total.")
    else:
        st.info("Pick at least one constructor.")

# ---- What drives the score (SHAP) ---------------------------------------
with tab_shap:
    st.subheader("What drives the score")
    s1, s2, s3 = st.columns(3)
    s1.metric("Podium classifier AUC-ROC", "0.928", "held-out 2024 season")
    s2.metric("Podium classifier Brier", "0.071", "well-calibrated", delta_color="off")
    s3.metric("BVI vs championship (Spearman ρ)", "0.718", "tracks, not a copy", delta_color="off")

    st.markdown(
        "**Starting grid position is the dominant feature for both models.** SHAP confirms "
        "for the Gradient Boosting points model and the calibrated podium classifier that "
        "where a car starts is the strongest driver of where it finishes — consistent with the "
        "Sprint 1 finding that qualifying gap-to-pole correlates with season points at "
        "Pearson r ≈ −0.79. That is exactly why the BVI weights Performance 60% and Consistency 40%."
    )

    any_fig = False
    cols = st.columns(2)
    for col, (title, fname) in zip(cols, SHAP_FIGURES.items()):
        with col:
            st.markdown(f"**{title}**")
            fig_path = find_figure(fname)
            if fig_path:
                st.image(fig_path, width='stretch')
                any_fig = True
            else:
                st.info(f"SHAP beeswarm `{fname}` not found — drop it in `reports/` to display it here.")
    if not any_fig:
        st.caption("Place the Sprint 3 SHAP beeswarm PNGs in a `reports/` folder next to this app to render them in-line.")

    st.success(
        "**The BVI tracks the championship but is not a copy of it** (Spearman ρ = 0.718). "
        "The clearest example: in 2018 Williams finished **10th (last)** on championship points "
        "but ranks **3rd** on the BVI that season — a slow car that converted its grid slots into "
        "reliable finishes, value a points-only table never sees."
    )

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.divider()
st.caption(
    "PitWall Intelligence · M.Sc. Data Science capstone, University of Europe for Applied Sciences · "
    "Dharmik Champaneri (20327984) · Supervisor: Dr. Humera Noor Minhas · "
    "Single data source: Jolpica-F1 API · scikit-learn · SHAP · Streamlit · Plotly."
)
