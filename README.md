# PitWall Intelligence

> **Predictive Modeling of Formula 1 Constructor Performance and Sponsorship Value Using Machine Learning**

![Status](https://img.shields.io/badge/status-Sprint%204%20%C2%B7%20dashboard-blue)
![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Capstone](https://img.shields.io/badge/M.Sc.-Capstone%202026-orange)

A data-driven Brand Value Index (BVI) for Formula 1 constructors, combining machine-learning performance prediction with SHAP-based explainability — quantifying what is currently a USD 1.8 billion sponsorship market priced largely on perception.

---

## Problem

Formula 1's sponsorship market exceeds USD 1.8 billion annually, with a global audience above 400 million viewers across broadcast, streaming, and digital channels. Yet sponsorship valuation in the sport still relies on brand-perception surveys, media-impression estimates, and subjective prestige scoring. There is no publicly available analytical framework that converts on-track performance into an interpretable, comparable, sponsor-facing metric.

## Research question

> Can explainability techniques applied to structured ML models trained on complete historical race data produce an interpretable composite score of F1 sponsorship value that discriminates between constructors?

## Brand Value Index (BVI)

Two-dimensional composite, season-normalised per constructor:

| Dimension | Weight | Components |
| --- | --- | --- |
| **Performance** | 60% | Predicted championship points · podium probability |
| **Consistency** | 40% | Reliability indicators · qualifying-to-race delta |

Min-max normalisation **within each season** ensures dominant-era seasons do not suppress midfield-team scores in cross-season comparison. Final score: `0.60 × Performance + 0.40 × Consistency`, scaled 0–100.

## Results so far

- **Sprint 1 — Qualifying pace anchors performance.** Average qualifying gap-to-pole correlates with total constructor points at **Pearson r ≈ −0.79** across 112 team-seasons (p < 0.001).
- **Sprint 2 — Models trained.** Baselines (Linear Regression, Decision Tree) plus advanced models: Gradient Boosting for points, and a Logistic Regression for podium probability.
- **Sprint 3 — BVI synthesised and explained.** The podium classifier is calibrated with Platt scaling and scores **AUC-ROC 0.928 / Brier 0.071** on a held-out 2024 season. SHAP attribution (TreeExplainer for the regressor, LinearExplainer for the classifier) shows **starting grid position is the dominant feature for both models**. The finished BVI tracks the championship at **Spearman ρ = 0.718** — close, but deliberately not a copy of the points table.
- **Sprint 4 — Interactive dashboard** (see below).

> One illustration of why the index is not just a re-skinned points table: in 2018 Williams finished **10th (last)** on championship points but ranks **3rd** on the BVI — a slow car that reliably converted its grid slots into finishes.

## Headline finding — Sprint 1

![Qualifying pace predicts championship points](https://github.com/DevDharmik/Pitwall-intelligence/raw/main/reports/eda_06_qual_vs_points.png)

Average qualifying gap-to-pole correlates with total constructor points at **Pearson r ≈ –0.79** across 112 team-seasons (p < 0.001). Qualifying pace anchors the BVI Performance dimension.

## Dataset

Single source — **[Jolpica-F1 API](https://api.jolpi.ca/ergast/)**, an actively maintained mirror of the Ergast Developer API for Formula 1. No Kaggle imports, no third-party aggregators, no synthetic data. Every record is fetched live and cached as JSON for reproducibility.

**Focal era:** V6 hybrid, 2014–2025 — stable technical regulations enabling like-for-like cross-season comparison. Analytical scope is fixed at **2014–2024**.

| Table | Rows | Coverage |
| --- | --- | --- |
| `races` | 228 | Grands Prix, 2014–2024 |
| `results` | 4,626 | Race finishing data |
| `qualifying` | 4,610 | Q1 / Q2 / Q3 session times |
| `constructor_standings` | 112 | Constructor-season finals |
| `driver_standings` | 247 | Driver-season finals |
| `constructors` · `drivers` | — | Team and driver metadata |

## Methodology

```
Jolpica API
  └─► ETL (requests + tenacity, JSON cache)
      └─► Preprocessing (DNF typing · qual parsing · gap-to-pole · season norm)
          └─► EDA (9 analyses)
              └─► Models
                  ├─ Baseline: Linear Regression · Decision Tree
                  └─ Advanced: Gradient Boosting · Logistic Regression + Platt
                      └─► Evaluation (5-fold CV RMSE / R² · AUC-ROC on held-out season)
                          └─► SHAP attribution
                              └─► BVI synthesis (Performance 60% + Consistency 40%)
                                  └─► Streamlit dashboard
```

## Dashboard (Sprint 4)

An interactive, sponsor-facing Streamlit app that wraps the BVI and its SHAP explanations into one tool. Pick any season (2014–2024) and explore:

- **Season standings** — ranked BVI bar chart and a table with each constructor's BVI, its Performance and Consistency components, championship rank and points side by side.
- **BVI map** — the full constructor-by-season heatmap, season-normalised on the 0–100 scale.
- **Performance vs Consistency** — a scatter of the two BVI dimensions with median quadrant lines (fast-but-fragile vs slow-but-dependable).
- **Constructor trajectory** — BVI across the era for any selection of teams.
- **What drives the score** — the SHAP explanation, the calibrated-classifier metrics, and the "most underrated / most flattered by the points table" headline for the chosen season.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads the season-normalised BVI table (`data/exports/bvi_scores.csv`) produced in Sprint 3 and the SHAP beeswarm PNGs from `reports/`. It retrains nothing — it consumes the Sprint 2–3 outputs directly.

## Tech stack

`Python 3.11` · `requests` · `tenacity` · `pandas` · `numpy` · `SQLite` · `scikit-learn` · `shap` · `matplotlib` · `seaborn` · `plotly` · `streamlit`

**Environment:** Google Colab · Jupyter · VS Code

## Sprint plan

| # | Window | Focus | Status |
| --- | --- | --- | --- |
| 1 | 23 Apr – 4 May | ETL pipeline · preprocessing · EDA | ✅ Complete |
| 2 | 5 May – 18 May | Baseline + advanced models | ✅ Complete |
| 3 | 19 May – 1 Jun | BVI synthesis · SHAP attribution | ✅ Complete |
| 4 | 2 Jun – 15 Jun | Streamlit dashboard | 🔵 In progress |
| 5 | 16 Jun – 22 Jun | Report polish · viva prep | ⚪ Planned |

**Final report due:** 22 June 2026 · **Defence:** 6 July 2026

## Repository structure

```
PitWall/
├── notebooks/
│   ├── 01_etl_jolpica.ipynb      # Jolpica → SQLite, JSON caching + retry
│   ├── 02_preprocessing.ipynb    # cleaning, feature engineering, season normalisation
│   ├── 03_eda.ipynb              # nine analyses driving Sprint 1 findings
│   ├── 04_features.ipynb         # team-season feature matrix
│   ├── 05_baseline_models.ipynb  # Linear Regression · Decision Tree
│   ├── 06_advanced_models.ipynb  # Gradient Boosting points model
│   └── 07_bvi_shap.ipynb         # podium classifier · SHAP · BVI synthesis
├── app.py                        # Streamlit dashboard (Sprint 4)
├── reports/                      # EDA + SHAP visual outputs (PNG, CSV)
├── data/
│   ├── pitwall.db                # SQLite store (gitignored)
│   ├── raw/                      # cached Jolpica JSON (gitignored)
│   └── exports/                  # bvi_scores.csv and other exports
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Reproducing

```bash
git clone https://github.com/DevDharmik/Pitwall-intelligence.git
cd Pitwall-intelligence
pip install -r requirements.txt
```

Open the notebooks in order (`01`→`07`) in Colab or Jupyter; they detect Colab vs a local environment automatically. `01_etl_jolpica.ipynb` populates `data/pitwall.db` from Jolpica on first run (cached JSON in `data/raw/` is reused afterwards), and `07_bvi_shap.ipynb` writes `data/exports/bvi_scores.csv`. Then launch the dashboard with `streamlit run app.py`.

End-to-end notebook runtime: ~10 minutes on a free Colab tier, dominated by initial Jolpica ingestion.

## Author

**Dharmik Champaneri** — Student ID 20327984
M.Sc. Data Science · University of Europe for Applied Sciences (Berlin / Potsdam)
**Supervisor:** Dr. Humera Noor Minhas
**Module:** Capstone Project · 2026

## License

[MIT License](LICENSE) for code. Reports and figures licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
