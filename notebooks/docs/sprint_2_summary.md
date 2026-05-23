# Sprint 2 — Baseline and Advanced ML Models

**Window:** 5 May 2026 – 18 May 2026
**Status:** Complete

## Goal

Train and evaluate machine-learning models to predict end-of-season constructor points from features available before the season ends. Compare a simple baseline against a stronger non-linear model.

## Deliverables

Three Jupyter notebooks in `notebooks/`:

- `04_features.ipynb` — engineers the `team_season_features` matrix (qualifying, reliability, and grid features) from the Sprint 1 tables.
- `05_baselines.ipynb` — Linear Regression and Decision Tree baselines.
- `06_advanced.ipynb` — Gradient Boosting Regressor, five-fold cross-validation, held-out season test, permutation importance, and the saved model artifact (`models/gbr_total_points_v1.joblib`).

## Methodology

### Target
Total constructor points per season — a continuous regression target.

### Features
Seven features from `team_season_features`: average grid position, average qualifying gap-to-pole, pole count, DNF rate, average qualifying-to-race delta, races entered, and season.

### Train / test split
Training: 2014–2023 (ten seasons). Held-out test: 2024 (one season). The held-out season is never seen during training or cross-validation.

### Models
- **Linear Regression** (standardised) — fast, interpretable baseline.
- **Decision Tree** (`max_depth=5`) — captures simple non-linear effects, baseline.
- **Gradient Boosting Regressor** (`scikit-learn`; 200 estimators, depth 3, learning rate 0.05) — advanced model, captures feature interactions.

### Evaluation
RMSE and R² on both five-fold cross-validation (training seasons) and on the held-out 2024 season. Permutation importance measures each feature's actual contribution beyond raw feature importance.

## Key findings

**Model performance** — metrics reflect the corrected `dnf_rate` feature (see the reliability fix in `04_features.ipynb`):

| Model              | CV R² (5-fold) | Held-out 2024 R² |
| ------------------ | -------------- | ---------------- |
| Linear Regression  | 0.934          | 0.956            |
| Decision Tree      | 0.898          | 0.851            |
| Gradient Boosting  | **0.950**      | **0.976**        |

Gradient Boosting is the best model, but the margin over Linear Regression is narrow — the linear baseline is itself strong (held-out R² 0.956). The advantage of Gradient Boosting shows most clearly on cross-validation and on error magnitude (RMSE), not as a dramatic R² gap. The Decision Tree generalises worst, with held-out R² well below its cross-validation score.

**Permutation importance:**
Average grid position dominates. Permuting grid position drops R² by roughly 1.09; permuting all six other features combined drops R² by only about 0.19.

In plain terms: where a constructor starts each race tells you almost everything about how it finishes the season. Race-day conversion, qualifying gap, and reliability still matter, but only at the margins. Qualifying pace is the spine.

This is not novel to people inside Formula 1, but watching it fall out of the data without being told to look for it is a useful sanity check on the pipeline.

## Caveat

The dominance of grid position is partly genuine and partly structural: grid position already encodes the underlying car pace that the other features depend on, and season points are themselves close to a deterministic function of finishing position. The high R² therefore reflects the model recovering known structure more than forecasting a hard outcome. The SHAP analysis in Sprint 3 makes this explicit — the project's analytical value comes from the Brand Value Index's Consistency dimension and the podium classifier, not from the points R² alone.

## Next steps

Sprint 3 adds:
- A Logistic Regression with Platt scaling for podium probability (classification target, complementing the regression).
- SHAP attribution on both the Gradient Boosting regressor and the podium classifier.
- The two-dimensional Brand Value Index — Performance (60 percent: predicted points + podium probability) and Consistency (40 percent: reliability indicators + qualifying-to-race delta).
