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
        result = "Correct" if predicted_mvp == actual_winner else "Missed"
        st.metric("Model's #1 Pick", predicted_mvp, delta=result)

st.divider()
st.subheader("What drives the model's predictions")
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
st.bar_chart(importance)

st.caption("Data: NBA Stats (1947-present) by sumitrodatta on Kaggle, sourced from Basketball-Reference.")
