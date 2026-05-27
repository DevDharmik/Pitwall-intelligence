# Sprint 3 — Brand Value Index, Podium Classifier, and SHAP Attribution

**Window:** 19 May 2026 – 1 June 2026
**Status:** Complete

## Goal

Turn the Sprint 2 points model into the project's headline deliverable: a two-dimensional Brand Value Index for Formula 1 constructors. This needs a second model the regression did not provide — a calibrated podium-probability classifier — and SHAP attribution so both models are interpretable rather than black boxes.

## Deliverables

One Jupyter notebook in `notebooks/`:

- `07_bvi_shap.ipynb` — reproduces the Sprint 2 Gradient Boosting points model, trains the podium classifier, runs SHAP attribution on both, and synthesises the Brand Value Index. Writes `bvi_scores.csv` and the `team_season_bvi` table.

## Methodology

### Podium probability classifier

A per-race-entry model: the probability that a car finishes on the podium. Logistic Regression with Platt scaling (`CalibratedClassifierCV`, sigmoid) so the output is a calibrated probability, not just a ranking. Three leakage-safe features, all known before the race starts: starting grid position, qualifying gap to pole, and the constructor's expanding-mean finishing position over prior rounds (`form_finish`). Trained on 2014–2023, tested on the held-out 2024 season; out-of-fold probabilities are generated for every entry as the honest BVI input.

### SHAP attribution

SHAP quantifies each feature's contribution to a single prediction. TreeExplainer is used on the Gradient Boosting points model — exact for tree ensembles. For the podium classifier, attribution is taken on the underlying standardised Logistic Regression via LinearExplainer; the Platt-scaling wrapper only rescales probabilities and does not change which features matter.

### Brand Value Index

Two dimensions, each the mean of two season-normalised components:

| Dimension   | Weight | Components                                              |
| ----------- | ------ | ------------------------------------------------------- |
| Performance | 60%    | predicted season points · mean podium probability       |
| Consistency | 40%    | reliability (`finish_rate`) · grid-to-finish stability  |

Conversion stability is the negative within-season standard deviation of the grid-to-finish delta — a team that turns its grid slot into a finishing position predictably scores high. Min-max normalisation is applied within each season so a dominant era does not crush midfield scores. BVI = 0.60·Performance + 0.40·Consistency, reported on a 0–100 scale.

## Key findings

**1. Points model.** Gradient Boosting scores R² 0.976 on the held-out 2024 season (RMSE 38.7). SHAP confirms the Sprint 2 permutation result: `avg_grid` dominates, with a mean absolute SHAP value roughly three times that of the next feature, `avg_qual_gap_to_pole`.

**2. Podium classifier.** Held-out 2024 performance: AUC-ROC 0.928, Brier score 0.071 — well ranked and well calibrated. SHAP shows starting grid drives podium log-odds most strongly, ahead of in-season form and qualifying gap.

**3. The BVI tracks the championship without copying it.** Mean Spearman ρ between BVI and championship points across 2014–2024 is 0.718. The gap from 1.0 is deliberate: the Consistency dimension and the podium model move teams off their raw points rank.

**4. Consistency is what separates the index from a points table.**

| Season | Constructor | Champ rank | BVI rank | Shift |
| ------ | ----------- | ---------- | -------- | ----- |
| 2018   | Williams    | 10         | 3        | +7    |
| 2019   | McLaren     | 4          | 10       | −6    |
| 2016   | McLaren     | 6          | 11       | −5    |
| 2015   | Sauber      | 8          | 3        | +5    |

2018 Williams scored almost nothing on points but converted its grid slots predictably, so the BVI rates it highly on Consistency. Whether that is the right signal for a sponsor is a question for the final report.

## Caveat

The Consistency dimension can reward a slow but reliable backmarker. This is visible in the divergence table and is not a bug to be patched silently — it is a genuine modelling choice. A sponsor may value a predictable, well-run team independently of raw results, or may not. The final report should argue the position explicitly rather than letting the weighting settle it quietly.

## Outputs

- `bvi_scores.csv` — BVI for all 112 team-seasons.
- `team_season_bvi` — the same scores as a database table.
- SHAP figures: `reports/sprint3_01_shap_gbr_beeswarm.png`, `sprint3_02_shap_gbr_bar.png`, `sprint3_03_shap_podium_beeswarm.png`.
- BVI heatmap: `reports/sprint3_04_bvi_heatmap.png`.

## Next steps

Sprint 4 builds the Streamlit dashboard: an interactive view of BVI by constructor and season, the SHAP attributions behind each score, and the championship-vs-BVI divergence.
