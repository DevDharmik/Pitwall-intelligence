# PitWall Intelligence

> **Predictive Modeling of Formula 1 Constructor Performance and Sponsorship Value Using Machine Learning**

![Status](https://img.shields.io/badge/status-Sprints_1--4_complete-success)
![Sprint](https://img.shields.io/badge/Sprint_5-report_%26_viva_prep-blue)
![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Dashboard](https://img.shields.io/badge/dashboard-live-success?logo=streamlit&logoColor=white)

A data-driven **Brand Value Index (BVI)** for Formula 1 constructors, combining machine-learning performance prediction with SHAP-based explainability — quantifying what is currently a USD 1.8 billion sponsorship market priced largely on perception.

**Live dashboard →** https://nnx9pdbkj7agwwsbn2m7bb.streamlit.app/

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

Min-max normalisation within each season ensures dominant-era seasons do not suppress midfield-team scores in cross-season comparison.

## Live dashboard

A Streamlit application (dark telemetry aesthetic) reads pre-computed exports and the trained model — no live API calls at view time, so it loads instantly and reproducibly. Five tabs:

| Tab | Purpose |
| --- | --- |
| **Grid** | Full BVI leaderboard, season-by-season |
| **Head-to-Head** | Constructor-vs-constructor comparison across the index |
| **Perf × Consistency** | The two BVI dimensions plotted against each other |
| **Evolution** | BVI trajectory of a constructor over the V6-hybrid era |
| **The Model** | SHAP attribution and model internals, made interpretable |

**Live:** https://nnx9pdbkj7agwwsbn2m7bb.streamlit.app/

## Headline results

**Constructor points model — GradientBoostingRegressor**

| Metric | Value |
| --- | --- |
| Test R² | 0.976 |
| 5-fold CV R² | 0.950 |
| RMSE | 38.7 |
| MAE | 29.0 |

**Explainability.** SHAP attribution is dominated by average grid position (`avg_grid`), accounting for ~72% of mean \|SHAP\| — qualifying pace is the single largest driver of predicted constructor points, confirming the Sprint 1 EDA signal.

**Podium probability — LogisticRegression + Platt calibration**

| Metric | Value |
| --- | --- |
| AUC-ROC | 0.928 |
| Brier score | 0.071 |

*(evaluated on a held-out season)*

**BVI validation.** McLaren tops the era at **85.6 / 100** (2024). Across 112 team-seasons, the BVI correlates with final championship rank at **Spearman ρ ≈ 0.72** — the index tracks competitive reality while remaining decomposable into its performance and consistency drivers.

**Supporting EDA finding (Sprint 1).** Average qualifying gap-to-pole correlates with total constructor points at **Pearson r ≈ –0.79** across 112 team-seasons (p < 0.001).

![Qualifying pace predicts championship points](https://github.com/DevDharmik/Pitwall-intelligence/raw/main/reports/eda_06_qual_vs_points.png)

> Metrics reflect committed notebook outputs (`reports/*.csv`); re-running notebooks `01 → 07` reproduces them end to end.

## Dataset

Single source — **[Jolpica-F1 API](https://api.jolpi.ca/ergast/)**, an actively maintained mirror of the Ergast Developer API for Formula 1. No Kaggle imports, no third-party aggregators, no synthetic data. Every record is fetched live and cached as JSON for reproducibility.

**Focal era:** V6 hybrid, 2014–2025 — stable technical regulations enabling like-for-like cross-season comparison.

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
 └─► ETL (requests + tenacity, JSON cache → SQLite)
   └─► Preprocessing (DNF typing · qual parsing · gap-to-pole · finish-rate allow-list · season norm)
     └─► EDA (9 analyses)
       └─► Feature engineering
         └─► Models
            ├─ Baseline: Linear Regression · Decision Tree
            └─ Advanced: Gradient Boosting (points) · Logistic Regression + Platt (podium)
           └─► Evaluation (5-fold CV R²/RMSE/MAE · AUC-ROC + Brier on held-out season)
             └─► SHAP attribution
               └─► BVI synthesis (Performance 60% + Consistency 40%, season min-max)
                 └─► Streamlit dashboard (5 tabs)
```

**Sprint 3 data-integrity correction.** `finish_rate` was initially constant at 1.0 because the source populates a finishing position for every classified driver, including retirements. Fixed with an explicit status allow-list (`Finished` / `Lapped` / `+N Lap(s)`), restoring a meaningful reliability signal for the Consistency dimension.

## Models & evaluation

| Stage | Model | Target | Headline |
| --- | --- | --- | --- |
| Baseline | Linear Regression, Decision Tree | Constructor points | Reference floor |
| Advanced | GradientBoostingRegressor | Constructor points | Test R² 0.976 · CV R² 0.950 |
| Advanced | LogisticRegression + Platt | Podium probability | AUC-ROC 0.928 · Brier 0.071 |
| Explainability | SHAP (TreeExplainer) | — | `avg_grid` ≈ 72% of mean \|SHAP\| |

Evaluation uses 5-fold cross-validation for the regressor and a held-out season for the calibrated classifier, avoiding within-season leakage.

## Tech stack

`Python 3.11` · `requests` · `tenacity` · `pandas` · `numpy` · `SQLite` · `scikit-learn` · `shap` · `matplotlib` · `seaborn` · `plotly` · `streamlit`

**Environment:** Google Colab · Jupyter · VS Code

## Sprint plan

| # | Window | Focus | Status |
| --- | --- | --- | --- |
| 1 | 23 Apr – 4 May | ETL pipeline · preprocessing · EDA | ✅ Complete |
| 2 | 5 May – 18 May | Baseline + advanced models | ✅ Complete |
| 3 | 19 May – 1 Jun | BVI synthesis · SHAP attribution | ✅ Complete |
| 4 | 2 Jun – 15 Jun | Streamlit dashboard | ✅ Complete |
| 5 | 16 Jun – 22 Jun | Report polish · viva prep | 🔄 In progress |

**Final report due:** 22 June 2026 · **Defence:** 6 July 2026

## Key findings by sprint

**Sprint 1 — EDA**
1. **Qualifying speed is the strongest single predictor of season points.** Pearson r ≈ –0.79 across 112 team-seasons (p < 0.001).
2. **The era is defined by sustained dominance.** Mercedes won eight consecutive Constructors' Championships (2014–2021), then Red Bull (2022, 2023) and McLaren (2024) — motivating within-season normalisation.
3. **Constructor rank volatility varies sharply.** Williams (σ = 2.83) and McLaren (σ = 2.44) are the most volatile; Force India (σ = 0.84), Red Bull (σ = 0.92), and Mercedes (σ = 1.04) the most stable — feeding the Consistency dimension.
4. **Season concentration ranges from 0.503 (2020, most competitive) to 0.619 (2016, most concentrated)**, era-wide mean Gini 0.556.

**Sprints 2–3 — modelling & synthesis**
5. Gradient Boosting lifts points prediction to Test R² 0.976 (CV R² 0.950) over the linear/tree baselines.
6. SHAP confirms qualifying pace (`avg_grid` ≈ 72%) as the dominant attributed feature — the EDA signal survives into the model.
7. The calibrated podium classifier reaches AUC-ROC 0.928 (Brier 0.071), feeding the Performance dimension as a probability rather than a binary.

**Sprint 4 — dashboard**
8. BVI operationalised as a live, five-tab Streamlit product; McLaren tops the era at 85.6/100, BVI–championship Spearman ρ ≈ 0.72.

## Repository structure

```
PitWall-intelligence/
├── notebooks/
│   ├── 01_etl_jolpica.ipynb       # Jolpica → SQLite, JSON caching + retry (tenacity)
│   ├── 02_preprocessing.ipynb     # cleaning, DNF typing, qual parsing, season normalisation
│   ├── 03_eda.ipynb               # nine analyses driving Sprint 1 findings
│   ├── 04_features.ipynb          # feature engineering for modelling
│   ├── 05_baselines.ipynb         # Linear Regression · Decision Tree
│   ├── 06_advanced.ipynb          # Gradient Boosting · Logistic Regression + Platt
│   └── 07_bvi_shap.ipynb          # BVI synthesis + SHAP attribution
├── dashboard/
│   └── app.py                     # Streamlit BVI dashboard (reads exports + saved model)
├── models/
│   └── gbr_points.joblib          # trained GradientBoosting model
├── reports/                       # EDA + model figures (PNG), metrics (CSV)
├── data/
│   ├── pitwall.db                 # SQLite store (gitignored)
│   ├── raw/                       # cached Jolpica JSON (gitignored)
│   └── exports/                   # pre-computed CSVs consumed by the dashboard
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Reproducing the project

```bash
git clone https://github.com/DevDharmik/Pitwall-intelligence.git
cd Pitwall-intelligence
pip install -r requirements.txt
```

**Notebooks** (run in order in Colab or Jupyter):
1. `01_etl_jolpica.ipynb` — populates `data/pitwall.db` from Jolpica; cached JSON in `data/raw/` is reused on later runs.
2. `02_preprocessing.ipynb` — builds the analytical tables.
3. `03_eda.ipynb` — generates the figures in `reports/`.
4. `04_features.ipynb` — assembles the modelling feature set.
5. `05_baselines.ipynb` / `06_advanced.ipynb` — train and evaluate the models; export metrics to `reports/`.
6. `07_bvi_shap.ipynb` — builds the BVI and SHAP attributions; writes the dashboard exports.

**Dashboard** (local):
```bash
streamlit run dashboard/app.py
```
End-to-end notebook runtime: ~10 minutes on a free Colab tier, dominated by initial Jolpica ingestion.

## Author

**Dharmik Champaneri** — Student ID 20327984
M.Sc. Data Science · University of Europe for Applied Sciences (Berlin / Potsdam)
**Supervisor:** Dr. Humera Noor Minhas
**Module:** Capstone Project · 2026

## License

[MIT License](https://github.com/DevDharmik/Pitwall-intelligence/blob/main/LICENSE) for code. Reports and figures licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
