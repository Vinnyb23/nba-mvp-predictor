NBA MVP Predictor 🏀
A machine learning project that predicts historical NBA Most Valuable Player winners from player and team statistics — built end-to-end in Python, from raw historical data to a deployed interactive app.
Live app: add your Streamlit Cloud URL here
Overview
Every season, the NBA MVP is decided by a media voting panel based on a mix of individual stats, team success, and narrative. This project asks: how well can a model trained purely on statistics reproduce those voting outcomes? A Random Forest model trained on seasons 1981–2025 correctly identifies the actual MVP winner as its #1 pick in 71% of seasons, using leave-one-season-out cross-validation.
Data source
NBA Stats (1947-present) by sumitrodatta on Kaggle — sourced from Basketball-Reference, covering:
•	Player Per Game.csv — per-game stats (points, assists, rebounds, shooting splits, etc.)
•	Advanced.csv — advanced metrics (PER, Win Shares, BPM, VORP, usage%, true shooting%)
•	Team Summaries.csv — team win/loss records and rating metrics
•	Player Award Shares.csv — actual historical MVP (and other award) voting results, including vote share and winner flag
Methodology
1.	Data merge: Player per-game and advanced stats were merged with team win percentage and MVP award share into one season-level player table. Players traded mid-season (multiple team rows in the source data) were consolidated into a single row using their combined-team stat line, tagged with the team they played the most games for.
2.	Scope: Filtered to the post-1981 modern voting era (MVP voting changed from a player poll to a media panel starting the 1980–81 season) and to legitimate candidates (at least 60% of that season's maximum games played, and 20+ minutes per game) — a threshold defined relative to season length so it correctly handles lockout-shortened seasons (1999, 2012) rather than excluding real MVP-caliber seasons.
3.	Target: MVP award share (0–1, continuous), so the model ranks candidates rather than just classifying a single winner.
4.	Models compared (single chronological holdout, train ≤2015 / test 2016–2025):
Model	Top-1 hit rate (10-season holdout)
Ridge Regression	50.0%
Random Forest	80.0%
XGBoost	80.0%

5.	Robust evaluation: Since a 10-season holdout is a small, potentially unrepresentative sample, both tree-based models were re-evaluated with leave-one-season-out cross-validation across all 45 seasons from 1981–2025 (train on all other seasons, predict the held-out one, repeat):
Model	Top-1 hit rate (45-season LOSO CV)
Random Forest	71.1% (32/45)
XGBoost	62.2% (28/45)

Random Forest was selected as the final model based on this more rigorous evaluation.
Features used
Per-game: points, assists, rebounds, steals, blocks, turnovers, FG%/3P%/FT%
Advanced: PER, true shooting%, usage%, offensive/defensive/total Win Shares, Win Shares per 48, offensive/defensive/total BPM, VORP
Team context: win percentage, Simple Rating System (SRS)
Other: age
VORP, Win Shares, BPM, and PER were the strongest individual correlates with MVP vote share — consistent with findings from other public MVP-prediction projects.
App
A Streamlit app lets you pick any season from 1981–2025 and see:
•	The model's top 5 predicted MVP candidates, ranked by predicted vote share
•	The actual MVP winner for that season, with a clear correct/missed indicator
•	Which stats the model weighs most heavily (feature importance chart)
Repository structure
nba-mvp-predictor/
├── data/
│   └── processed/
│       └── mvp_model_data.csv   # cleaned, modeling-ready season-level table
├── notebooks/                    # data prep, EDA, and modeling notebooks (Colab)
├── src/                          # reusable data prep / feature / training scripts
├── app/
│   └── streamlit_app.py          # the deployed Streamlit app
├── models/
│   ├── rf_mvp_model.joblib        # trained Random Forest model
│   └── feature_cols.joblib        # feature list used at inference time
├── requirements.txt
└── README.md

Running locally
pip install -r requirements.txt
streamlit run app/streamlit_app.py

Limitations & future work
•	The model only uses box-score-derived stats; it can't capture narrative factors (media storylines, "turn"/fatigue effects, close-race voter fatigue) that occasionally swing real voting, which explains most of its misses (e.g., 2018, 2019, 2020, 2023, 2025).
•	Potential next steps: live in-season prediction using current-year stats, hyperparameter tuning, and incorporating additional context like strength of schedule or clutch-performance stats.
Acknowledgments
Built as a personal learning project to understand end-to-end applied machine learning: data acquisition, feature engineering, model comparison, rigorous cross-validation, and deployment.
