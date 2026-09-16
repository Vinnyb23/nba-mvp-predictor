import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="NBA MVP Predictor", page_icon="🏀", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("models/rf_mvp_model.joblib")
    feature_cols = joblib.load("models/feature_cols.joblib")
    return model, feature_cols

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/mvp_model_data.csv")

model, feature_cols = load_model()
df = load_data()

st.title("🏀 NBA MVP Predictor")
st.write("Predicting historical NBA MVP winners from player stats using a Random Forest model trained on seasons 1981-2025 (leave-one-season-out validated hit rate: ~71%).")

st.divider()
st.subheader("2026-27 Season: Real Market Odds")
st.caption("The model above predicts historical MVP winners from completed-season stats. The 2026-27 season hasn't started yet, so there's no season-average data for it — instead, here's what real sportsbooks currently think, as a preview of who's favored.")

odds_df = pd.read_csv("data/mvp_odds.csv")
latest_date = odds_df["as_of_date"].max()
latest_odds = odds_df[odds_df["as_of_date"] == latest_date].sort_values("consensus_implied_prob", ascending=False)

st.bar_chart(latest_odds.set_index("player")["consensus_implied_prob"])
st.dataframe(
    latest_odds[["player", "consensus_odds_american", "consensus_implied_prob"]]
    .rename(columns={
        "player": "Player",
        "consensus_odds_american": "Consensus Odds",
        "consensus_implied_prob": "Implied Win Probability"
    })
    .set_index("Player"),
    use_container_width=True
)
st.caption(f"Odds as of {latest_date}, averaged across DraftKings, FanDuel, BetMGM, Caesars, and ESPN BET futures markets. Snapshot refreshed periodically, not live.")

seasons = sorted(df['season'].unique(), reverse=True)
selected_season = st.selectbox("Select a season", seasons)

season_df = df[df['season'] == selected_season].copy()
X = season_df[feature_cols].fillna(0)
season_df['predicted_share'] = model.predict(X)

top5 = season_df.sort_values('predicted_share', ascending=False).head(5)
actual_winner_row = season_df[season_df['winner'] == True]
actual_winner = actual_winner_row['player'].values[0] if len(actual_winner_row) > 0 else "Not yet available"

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Model's Top 5 Predicted MVP Candidates")
    display_cols = ['player', 'team', 'predicted_share', 'pts_per_game', 'ws', 'vorp', 'win_pct']
    st.dataframe(
        top5[display_cols].rename(columns={
            'predicted_share': 'Predicted Share', 'pts_per_game': 'PPG',
            'ws': 'Win Shares', 'vorp': 'VORP', 'win_pct': 'Team Win%'
        }).reset_index(drop=True),
        use_container_width=True
    )

with col2:
    st.subheader("Actual Result")
    st.metric("Real MVP Winner", actual_winner)
    predicted_mvp = top5.iloc[0]['player']
    if actual_winner != "Not yet available":
        if predicted_mvp == actual_winner:
            st.success(f"✅ Correct — model's #1 pick ({predicted_mvp}) matches the actual winner.")
        else:
            st.error(f"❌ Missed — model picked {predicted_mvp}, actual winner was {actual_winner}.")
    else:
        st.info("Actual result not yet available for this season.")

st.divider()
st.subheader("What drives the model's predictions")
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
st.bar_chart(importance)

GLOSSARY = {
    "age": "Player's age during that season",
    "pts_per_game": "Points per game",
    "ast_per_game": "Assists per game",
    "trb_per_game": "Total rebounds per game",
    "stl_per_game": "Steals per game",
    "blk_per_game": "Blocks per game",
    "tov_per_game": "Turnovers per game",
    "fg_percent": "Field goal percentage",
    "x3p_percent": "Three-point field goal percentage",
    "ft_percent": "Free throw percentage",
    "per": "Player Efficiency Rating — all-in-one per-minute productivity rating, league average = 15",
    "ts_percent": "True Shooting % — shooting efficiency across 2s, 3s, and free throws combined",
    "usg_percent": "Usage % — estimated share of team plays used by a player while on the floor",
    "ows": "Offensive Win Shares — estimated wins contributed through offense",
    "dws": "Defensive Win Shares — estimated wins contributed through defense",
    "ws": "Win Shares — total estimated wins contributed (offense + defense)",
    "ws_48": "Win Shares per 48 minutes — Win Shares rate normalized to a full game",
    "obpm": "Offensive Box Plus/Minus — offensive points per 100 possessions above a league-average player",
    "dbpm": "Defensive Box Plus/Minus — same, for defense",
    "bpm": "Box Plus/Minus — total contribution per 100 possessions above a league-average player",
    "vorp": "Value Over Replacement Player — total points contributed above a replacement-level player, prorated to an 82-game season",
    "win_pct": "Team's winning percentage that season",
    "srs": "Simple Rating System — team rating based on point differential adjusted for strength of schedule",
}

with st.expander("What do these stats mean?"):
    glossary_df = pd.DataFrame(
        [(feat, GLOSSARY.get(feat, "")) for feat in importance.index],
        columns=["Stat", "Meaning"]
    )
    st.table(glossary_df.set_index("Stat"))

st.caption("Data: NBA Stats (1947-present) by sumitrodatta on Kaggle, sourced from Basketball-Reference.")
