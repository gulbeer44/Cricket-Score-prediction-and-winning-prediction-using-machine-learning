import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt
import os, joblib

# =========================
# STREAMLIT PAGE SETUP
# =========================
st.set_page_config(page_title="IPL Prediction Suite", layout="centered")
st.title("🏏 IPL Prediction Suite")

# ======================================================
# TAB 1: FIRST INNINGS SCORE PREDICTION (ipl.csv)
# ======================================================
@st.cache_data(show_spinner=False)
def load_ipl_first_innings_data():
    ipl = pd.read_csv("ipl.csv")

    required_cols = ['bat_team', 'bowl_team', 'venue', 'runs', 'wickets', 'overs', 'total']
    missing = [c for c in required_cols if c not in ipl.columns]
    if missing:
        raise ValueError(f"ipl.csv is missing columns: {missing}. Found: {list(ipl.columns)}")

    ipl = ipl[required_cols].dropna()
    for c in ['runs', 'wickets', 'overs', 'total']:
        ipl[c] = pd.to_numeric(ipl[c], errors='coerce')
    ipl = ipl.dropna(subset=['runs', 'wickets', 'overs', 'total'])

    return ipl


@st.cache_data(show_spinner=False)
def train_first_innings_model(ipl_df: pd.DataFrame):
    X = ipl_df[['bat_team', 'bowl_team', 'venue', 'runs', 'wickets', 'overs']]
    y = ipl_df['total']

    categorical_cols = ['bat_team', 'bowl_team', 'venue']
    numeric_cols = ['runs', 'wickets', 'overs']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', 'passthrough', numeric_cols),
        ],
        remainder='drop'
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    model.fit(X, y)

    teams_bat = sorted(X['bat_team'].unique().tolist())
    teams_bowl = sorted(X['bowl_team'].unique().tolist())
    venues = sorted(X['venue'].unique().tolist())
    return model, teams_bat, teams_bowl, venues


# ======================================================
# TAB 2: WIN PROBABILITY (matches.csv + deliveries.csv)
# ======================================================
@st.cache_data(show_spinner=False)
def load_and_prepare_winprob_data():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    first_innings_totals = (
        deliveries.loc[deliveries['inning'] == 1]
        .groupby('match_id', as_index=False)['total_runs']
        .sum()
        .rename(columns={'total_runs': 'target_runs'})
    )

    chase_df = deliveries.loc[deliveries['inning'] == 2].copy()
    df = (
        chase_df
        .merge(matches[['id', 'city', 'winner']], left_on='match_id', right_on='id', how='left')
        .merge(first_innings_totals, on='match_id', how='left')
    )

    if 'batting_team' not in df.columns or 'bowling_team' not in df.columns:
        raise ValueError("deliveries.csv must contain 'batting_team' and 'bowling_team' columns.")

    df = df.sort_values(['match_id', 'over', 'ball']).reset_index(drop=True)
    df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()
    df['balls_done'] = (df['over'] - 1) * 6 + df['ball']
    df['balls_left'] = 120 - df['balls_done']
    df['runs_left'] = df['target_runs'] - df['current_score']
    df['cum_wkts'] = df.groupby('match_id')['player_dismissed'].transform(lambda s: s.notna().cumsum())
    df['wickets'] = 10 - df['cum_wkts']

    df['balls_done_nz'] = df['balls_done'].replace(0, np.nan)
    df['balls_left_nz'] = df['balls_left'].replace(0, np.nan)
    df['crr'] = (df['current_score'] * 6) / df['balls_done_nz']
    df['rrr'] = (df['runs_left'] * 6) / df['balls_left_nz']

    clean = df[
        df['target_runs'].notna()
        & df['balls_done'].gt(0)
        & df['balls_left'].ge(0)
    ].copy()

    clean['result'] = (clean['winner'] == clean['batting_team']).astype(int)

    final_data = clean[[
        'batting_team', 'bowling_team', 'city',
        'runs_left', 'balls_left', 'wickets',
        'target_runs', 'crr', 'rrr', 'result'
    ]].dropna().fillna(0)

    for c in ['runs_left', 'balls_left', 'wickets', 'target_runs', 'crr', 'rrr', 'result']:
        final_data[c] = pd.to_numeric(final_data[c], errors='coerce')

    return final_data


@st.cache_data(show_spinner=False)
def train_winprob_model(final_data: pd.DataFrame):
    X = final_data.drop(columns=['result'])
    y = final_data['result']

    categorical_features = ['batting_team', 'bowling_team', 'city']
    numeric_features = ['runs_left', 'balls_left', 'wickets', 'target_runs', 'crr', 'rrr']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', 'passthrough', numeric_features),
        ],
        remainder='drop'
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', max_iter=300, random_state=42))
    ])
    model.fit(X, y)

    teams = sorted(pd.unique(pd.concat([X['batting_team'], X['bowling_team']], ignore_index=True)).tolist())
    cities = sorted(X['city'].dropna().unique().tolist())
    return model, teams, cities


# ============================
# BUILD THE UI WITH TWO TABS
# ============================
tab1, tab2 = st.tabs(["First Innings Score Prediction", "Win Probability Prediction"])

# ------------- TAB 1 -------------
with tab1:
    st.header("First Innings Score Prediction")

    try:
        ipl_df = load_ipl_first_innings_data()
        score_model, bat_choices, bowl_choices, venue_choices = train_first_innings_model(ipl_df)

        col1, col2 = st.columns(2)
        with col1:
            bat_team = st.selectbox("Batting Team", bat_choices, index=0)
        with col2:
            bowl_team = st.selectbox("Bowling Team", bowl_choices, index=0)

        venue = st.selectbox("Venue", venue_choices, index=0)
        runs = st.number_input("Runs scored so far", min_value=0, value=50)
        wickets = st.number_input("Wickets fallen", min_value=0, max_value=10, value=2)
        overs = st.number_input("Overs completed", min_value=0.0, max_value=20.0, value=10.0, step=0.1)

        if st.button("Predict 1st Innings Score"):
            if overs >= 20.0 or wickets >= 10:
                st.success(f"Innings finished. Final Score: {runs}")
            else:
                inp = pd.DataFrame([{
                    'bat_team': bat_team,
                    'bowl_team': bowl_team,
                    'venue': venue,
                    'runs': runs,
                    'wickets': wickets,
                    'overs': overs
                }])
                pred = score_model.predict(inp)[0]
                st.success(f"Predicted First Innings Score: {int(round(pred))}")

    except Exception as e:
        st.error(f"[Score Prediction] {e}")

# ------------- TAB 2 -------------
with tab2:
    st.header("Win Probability Prediction")

    try:
        final_data = load_and_prepare_winprob_data()
        win_model, team_choices, city_choices = train_winprob_model(final_data)

        col1, col2 = st.columns(2)
        with col1:
            batting_team = st.selectbox('Batting team (chasing)', team_choices, index=0)
        with col2:
            bowling_team = st.selectbox('Bowling team', team_choices, index=0)

        selected_city = st.selectbox('City', city_choices, index=0)
        target = st.number_input('Target (1st innings total)', min_value=0, value=150)

        col3, col4, col5 = st.columns(3)
        with col3:
            score = st.number_input('Current score', min_value=0, value=100)
        with col4:
            wickets_out = st.number_input('Wickets out', min_value=0, max_value=9, value=3)
        with col5:
            overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, value=12.0, step=0.1)

        if st.button('Predict Win Probability'):
            runs_left = target - score
            balls_done = int(round(overs * 6))
            balls_left = max(0, 120 - balls_done)
            wickets_remaining = max(0, 10 - wickets_out)

            if runs_left <= 0:
                st.success(f"{batting_team} has already won! ✅")
            elif balls_left == 0 and runs_left > 0:
                st.success(f"{bowling_team} has already won! ✅")
            else:
                crr = (score * 6 / balls_done) if balls_done > 0 else 0.0
                rrr = (runs_left * 6 / balls_left) if balls_left > 0 else np.inf

                df = pd.DataFrame([{
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,
                    'city': selected_city,
                    'runs_left': runs_left,
                    'balls_left': balls_left,
                    'wickets': wickets_remaining,
                    'target_runs': target,
                    'crr': crr,
                    'rrr': rrr
                }]).replace([np.inf, -np.inf], np.nan).fillna(0)

                proba = win_model.predict_proba(df)[0]
                batting_win = int(round(proba[1] * 100))
                bowling_win = 100 - batting_win

                st.subheader("Winning Probability")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric(f"{batting_team}", f"{batting_win}%")
                    st.progress(batting_win / 100)
                with c2:
                    st.metric(f"{bowling_team}", f"{bowling_win}%")
                    st.progress(bowling_win / 100)

                st.caption(f"CRR: {crr:.2f} | RRR: {0 if np.isinf(rrr) else rrr:.2f} | Runs left: {runs_left} | Balls left: {balls_left} | Wkts in hand: {wickets_remaining}")

    except Exception as e:
        st.error(f"[Win Probability] {e}")

# ======================================================
# 🌍 OTHER FORMATS (T20 | ODI | TEST)
# ======================================================
st.markdown("---")
st.header("🌍 Other Formats: T20 | ODI | Test Predictions")

@st.cache_data(show_spinner=False)
def load_format_csv(format_name):
    csv_path = f"{format_name}.csv"
    if not os.path.exists(csv_path):
        st.warning(f"{csv_path} not found.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    rename_map = {
        'batting_team': 'bat_team',
        'bowling_team': 'bowl_team',
        'ground': 'venue',
        'stadium': 'venue',
        'score': 'runs',
        'wkts': 'wickets',
        'total_runs': 'total'
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    expected = ['bat_team', 'bowl_team', 'venue', 'runs', 'wickets', 'overs', 'total']
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan
    df = df.dropna(subset=['bat_team', 'bowl_team', 'total'])
    return df[expected]


def train_model_for_format(df, format_name, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{format_name}_model.joblib")

    for c in ['runs', 'wickets', 'overs', 'total']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['runs', 'wickets', 'overs', 'total'])

    # Simulate realistic mid-innings (25%, 50%, 75%) progression like IPL
    new_rows = []
    max_overs = 20 if "t20" in format_name else 50 if "odi" in format_name else 90
    for _, row in df.iterrows():
        total = row['total']
        for p in [0.25, 0.5, 0.75]:
            overs_p = round(max_overs * p, 2)
            runs_p = (total / max_overs) * overs_p
            wkts_p = min(10, int(row['wickets'] * p + 0.5))
            new_rows.append({
                'bat_team': row['bat_team'],
                'bowl_team': row['bowl_team'],
                'venue': row['venue'],
                'runs': runs_p,
                'wickets': wkts_p,
                'overs': overs_p,
                'total': total
            })
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    X = df[['bat_team', 'bowl_team', 'venue', 'runs', 'wickets', 'overs']]
    y = df['total']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['bat_team', 'bowl_team', 'venue']),
        ('num', 'passthrough', ['runs', 'wickets', 'overs'])
    ])
    model = Pipeline([
        ('pre', preprocessor),
        ('reg', LinearRegression())
    ])
    model.fit(X, y)
    preds = model.predict(X)
    rmse = sqrt(mean_squared_error(y, preds))
    joblib.dump(model, model_path)

    return model, sorted(X['bat_team'].unique()), sorted(X['bowl_team'].unique()), sorted(X['venue'].unique()), rmse, model_path


formats = {
    "t20s": {"label": "🏏 T20 Prediction", "max_overs": 20.0, "default_runs": 50},
    "odis": {"label": "🏆 ODI Prediction", "max_overs": 50.0, "default_runs": 150},
    "tests": {"label": "🧢 Test Prediction", "max_overs": 90.0, "default_runs": 200}
}

tabs = st.tabs([formats[f]["label"] for f in formats])

for i, (fmt, cfg) in enumerate(formats.items()):
    with tabs[i]:
        st.subheader(cfg["label"].split(" ")[1] + " First Innings Score Prediction")
        df = load_format_csv(fmt)
        if df.empty:
            st.info(f"{fmt.upper()} data missing or empty.")
            continue
        model, bats, bowls, venues, rmse, path = train_model_for_format(df, fmt)
        st.caption(f"Model saved: {path} | RMSE: {rmse:.2f}")

        bat = st.selectbox("Batting Team", bats, key=f"bat_{fmt}")
        bowl = st.selectbox("Bowling Team", bowls, key=f"bowl_{fmt}")
        venue = st.selectbox("Venue", venues, key=f"venue_{fmt}")
        runs = st.number_input("Runs so far", min_value=0, value=cfg["default_runs"], key=f"runs_{fmt}")
        wickets = st.number_input("Wickets fallen", min_value=0, max_value=10, value=2, key=f"wkt_{fmt}")
        overs = st.number_input("Overs completed", min_value=0.0, max_value=cfg["max_overs"],
                                value=cfg["max_overs"] / 2, step=0.1, key=f"ov_{fmt}")

        if st.button(f"Predict {fmt.upper()} Score", key=f"btn_{fmt}"):
            inp = pd.DataFrame([{
                'bat_team': bat,
                'bowl_team': bowl,
                'venue': venue,
                'runs': runs,
                'wickets': wickets,
                'overs': overs
            }])
            pred = model.predict(inp)[0]
            st.success(f"Predicted {fmt.upper()} First Innings Score: {int(round(pred))}")
