# NBA MVP Predictor 🏀

A machine learning project that predicts historical NBA Most Valuable Player winners from player and team statistics — built end-to-end in Python, from raw historical data to a deployed interactive app.

**Live app:** (https://nba-mvp-predictor-5k83qxa87cdfxnuxphravg.streamlit.app/)

## Overview

Every season, the NBA MVP is decided by a media voting panel based on a mix of individual stats, team success, and narrative. This project asks: how well can a model trained purely on statistics reproduce those voting outcomes? A Random Forest model trained on seasons 1981–2025 correctly identifies the actual MVP winner as its #1 pick in **71% of seasons**, using leave-one-season-out cross-validation.

## Data source

[NBA Stats (1947-present)](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats) by sumitrodatta on Kaggle — sourced from Basketball-Reference, covering:
- `Player Per Game.csv` — per-game stats (points, assists, rebounds, shooting splits, etc.)
- `Advanced.csv` — advanced metrics (PER, Win Shares, BPM, VORP, usage%, true shooting%)
- `Team Summaries.csv` — team win/loss records and rating metrics
- `Player Award Shares.csv` — actual historical MVP (and other award) voting results, including vote share and winner flag

## Methodology

1. **Data merge:** Player per-game and advanced stats were merged with team win percentage and MVP award share into one season-level player table. Players traded mid-season (multiple team rows in the source data) were consolidated into a single row using their combined-team stat line, tagged with the team they played the most games for.
2. **Scope:** Filtered to the post-1981 modern voting era (MVP voting changed from a player poll to a media panel starting the 1980–81 season) and to legitimate candidates (at least 60% of that season's maximum games played, and 20+ minutes per game) — a threshold defined relative to season length so it correctly handles lockout-shortened seasons (1999, 2012) rather than excluding real MVP-caliber seasons.
3. **Target:** MVP award share (0–1, continuous), so the model ranks candidates rather than just classifying a single winner.
4. **Models compared** (single chronological holdout, train ≤2015 / test 2016–2025):

   | Model | Top-1 hit rate (10-season holdout) |
   |---|---|
   | Ridge Regression | 50.0% |
   | Random Forest | 80.0% |
   | XGBoost | 80.0% |

5. **Robust evaluation:** Since a 10-season holdout is a small, potentially unrepresentative sample, both tree-based models were re-evaluated with leave-one-season-out cross-validation across all 45 seasons from 1981–2025 (train on all other seasons, predict the held-out one, repeat):

   | Model | Top-1 hit rate (45-season LOSO CV) |
   |---|---|
   | Random Forest | **71.1%** (32/45) |
   | XGBoost | 62.2% (28/45) |

   Random Forest was selected as the final model based on this more rigorous evaluation.

## Hyperparameter tuning

The Random Forest was tuned using `RandomizedSearchCV` (40 candidates x 5 folds, grouped by season with `GroupKFold` so no season's rows leaked across train/validation splits within the search) on the training seasons only, optimizing for RMSE:

| | Default RF | Tuned RF |
|---|---|---|
| Params | `n_estimators=300, max_depth=6` | `n_estimators=300, max_depth=8, min_samples_split=5, min_samples_leaf=4, max_features=1.0` |
| Holdout top-1 hit rate (2016-2025) | 80.0% (8/10) | 80.0% (8/10) |
| **LOSO top-1 hit rate (1981-2025, 45 seasons)** | **71.1% (32/45)** | 68.9% (31/45) |

The tuned model achieved a lower training-fold RMSE but a *worse* LOSO top-1 hit rate than the untuned default model. This is a useful reminder that RMSE (how close predicted vote share is on average) and top-1 ranking accuracy (whether the single highest-predicted player is the actual winner) aren't the same objective — optimizing for one doesn't guarantee improving the other, especially with a small dataset (~45 MVP seasons) where estimates carry meaningful noise. **The default-parameter Random Forest was kept as the final model** based on this more rigorous, task-relevant evaluation.

## Features used

Per-game: points, assists, rebounds, steals, blocks, turnovers, FG%/3P%/FT%
Advanced: PER, true shooting%, usage%, offensive/defensive/total Win Shares, Win Shares per 48, offensive/defensive/total BPM, VORP
Team context: win percentage, Simple Rating System (SRS)
Other: age

VORP, Win Shares, BPM, and PER were the strongest individual correlates with MVP vote share — consistent with findings from other public MVP-prediction projects.

## App

A Streamlit app lets you pick any season from 1981–2025 and see:
- The model's top 5 predicted MVP candidates, ranked by predicted vote share
- The actual MVP winner for that season, with a clear correct/missed indicator
- Which stats the model weighs most heavily (feature importance chart)

## Repository structure
