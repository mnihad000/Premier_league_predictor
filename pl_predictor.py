# pl_epl_predictor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


CSV_PATH = "C:/projects/premier_league_predictor/matches (1).csv"    # your FBRef team log CSV
FILTER_COMP = "Premier League"  # your file is Prem-only; keeping this is fine
TARGET_MODE = "multiclass"      # H/D/A
TEST_START_DATE = "2022-01-01"  # train < this, test >= this
ROLL_WINDOW = 5                 # rolling form window (matches)


def teamlog_to_matchlevel(df_team: pd.DataFrame, filter_comp: str | None) -> pd.DataFrame:
    df = df_team.copy()

    # (optional) keep only Premier League
    if filter_comp is not None and "comp" in df.columns:
        df = df[df["comp"] == filter_comp].copy()

    # only rows clearly marked Home/Away
    df = df[df["venue"].isin(["Home", "Away"])].copy()

    # handle match id column name: "match report" (FBRef) or "match_report"
    match_id = "match report" if "match report" in df.columns else (
        "match_report" if "match_report" in df.columns else None
    )
    if match_id is None:
        raise KeyError("Could not find a 'match report' (or 'match_report') column to join on.")

    # HOME rows (one per match from home perspective)
    home = df[df["venue"] == "Home"][[
        match_id, "date", "time", "season",
        "team", "opponent", "gf", "ga", "sh", "sot", "xg"
    ]].rename(columns={
        "team": "home_team",
        "opponent": "away_team",
        "gf": "home_goals",
        "ga": "away_goals",
        "sh": "home_shots",
        "sot": "home_shots_on_target",
        "xg": "home_xg"
    })

    # AWAY rows (do NOT include away_goals here to avoid duplicate columns)
    away = df[df["venue"] == "Away"][[
        match_id, "team", "opponent", "sh", "sot", "xg"
    ]].rename(columns={
        "team": "away_team",
        "opponent": "home_team",  # away row lists opponent = home team
        "sh": "away_shots",
        "sot": "away_shots_on_target",
        "xg": "away_xg"
    })

    # Merge by unique match identifier + team pairing
    df_ml = home.merge(
        away[[match_id, "home_team", "away_team", "away_shots", "away_shots_on_target", "away_xg"]],
        on=[match_id, "home_team", "away_team"],
        how="inner"
    )

    # Parse & clean
    df_ml["date"] = pd.to_datetime(df_ml["date"], errors="coerce")
    df_ml = df_ml.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    df_ml = df_ml.sort_values("date").reset_index(drop=True)
    return df_ml


def add_outcome_multiclass(df: pd.DataFrame) -> pd.DataFrame:
    df["outcome"] = np.where(
        df["home_goals"] > df["away_goals"], "H",
        np.where(df["home_goals"] < df["away_goals"], "A", "D")
    )
    return df


def build_team_long(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["date", "home_team", "away_team", "home_goals", "away_goals",
                 "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
                 "home_xg", "away_xg"]

    # ensure optional columns exist
    for c in ["home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target", "home_xg", "away_xg"]:
        if c not in df.columns:
            df[c] = np.nan

    home = df[base_cols].copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["gf"] = home["home_goals"]
    home["ga"] = home["away_goals"]
    home["is_home"] = 1

    away = df[base_cols].copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["gf"] = away["away_goals"]
    away["ga"] = away["home_goals"]
    away["is_home"] = 0

    def take_side(df_side):
        out = df_side[["date", "team", "opponent", "is_home", "gf", "ga"]].copy()
        out["for_shots"] = np.where(df_side["is_home"] == 1, df_side["home_shots"], df_side["away_shots"])
        out["against_shots"] = np.where(df_side["is_home"] == 1, df_side["away_shots"], df_side["home_shots"])
        out["for_sot"] = np.where(df_side["is_home"] == 1, df_side["home_shots_on_target"], df_side["away_shots_on_target"])
        out["against_sot"] = np.where(df_side["is_home"] == 1, df_side["away_shots_on_target"], df_side["home_shots_on_target"])
        out["for_xg"] = np.where(df_side["is_home"] == 1, df_side["home_xg"], df_side["away_xg"])
        out["against_xg"] = np.where(df_side["is_home"] == 1, df_side["away_xg"], df_side["home_xg"])
        return out

    team_long = pd.concat([take_side(home), take_side(away)], ignore_index=True)
    return team_long

def add_rolling_form(team_long: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add rolling form features using only prior matches (no leakage)."""
    team_long = team_long.sort_values(["team", "date"]).copy()

    # Basic outcomes
    team_long["points"] = np.select(
        [team_long["gf"] > team_long["ga"], team_long["gf"] == team_long["ga"]],
        [3, 1], default=0
    )
    team_long["win"]  = (team_long["gf"] > team_long["ga"]).astype(int)
    team_long["draw"] = (team_long["gf"] == team_long["ga"]).astype(int)
    team_long["gd"]   = team_long["gf"] - team_long["ga"]  # goal difference

    # Roll these as means over the last N prior matches
    roll_cols = [
        "gf", "ga", "gd",
        "for_shots", "against_shots",
        "for_sot", "against_sot",
        "for_xg", "against_xg",
        "points", "win"
    ]

    for c in roll_cols:
        team_long[f"roll_{c}_{window}"] = (
            team_long.groupby("team")[c]
                     .shift(1)  # exclude current match → no leakage
                     .rolling(window=window, min_periods=window)
                     .mean()
                     .values
        )

    # Explicit draw rate (separate name so it's easy to spot/use)
    team_long[f"roll_draw_rate_{window}"] = (
        team_long.groupby("team")["draw"]
                 .shift(1)
                 .rolling(window=window, min_periods=window)
                 .mean()
                 .values
    )

    return team_long


def reattach_to_matches(df_matches: pd.DataFrame, team_long_with_rolls: pd.DataFrame) -> pd.DataFrame:
    # home features
    home_feats = team_long_with_rolls.rename(columns=lambda c: f"home_{c}")
    merged = df_matches.merge(
        home_feats,
        left_on=["date", "home_team"],
        right_on=["home_date", "home_team"],
        how="left"
    ).drop(columns=["home_date"])

    # away features
    away_feats = team_long_with_rolls.rename(columns=lambda c: f"away_{c}")
    merged = merged.merge(
        away_feats,
        left_on=["date", "away_team"],
        right_on=["away_date", "away_team"],
        how="left"
    ).drop(columns=["away_date"])

    return merged

df_team = pd.read_csv(CSV_PATH)

# build match-level
df = teamlog_to_matchlevel(df_team, FILTER_COMP)

# labels: H/D/A
df = add_outcome_multiclass(df)

# rolling form
team_long = build_team_long(df)
team_long = add_rolling_form(team_long, ROLL_WINDOW)
df_feat = reattach_to_matches(df, team_long)

# features
cat_features = ["home_team", "away_team"]
roll_features = [c for c in df_feat.columns if c.startswith("home_roll_") or c.startswith("away_roll_")]
num_features = roll_features  # (add non-leaky numeric features if you want)

# time-based split
test_start = pd.Timestamp(TEST_START_DATE)
train_df = df_feat[df_feat["date"] < test_start].copy()
test_df  = df_feat[df_feat["date"] >= test_start].copy()

# targets
y_train = train_df["outcome"]
y_test  = test_df["outcome"]

# preprocess
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), num_features),
    ],
    remainder="drop"
)

# model
clf = RandomForestClassifier(
    n_estimators=600,
    min_samples_split=10,
    class_weight={"H":1.0, "D":2.0, "A":1.0},  # try also "balanced"
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([("prep", preprocess), ("clf", clf)])

# train
pipe.fit(train_df[cat_features + num_features], y_train)

# evaluate
preds = pipe.predict(test_df[cat_features + num_features])
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="macro")
print(f"Test Accuracy: {acc:.3f}")
print(f"Macro F1:      {f1:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, preds, digits=3))
print("Confusion Matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, preds))

last = df.sort_values("date").iloc[-1]
home_name = last["home_team"]
away_name = last["away_team"]

# If you already have latest_roll_for() defined above, you can reuse it.
# If not, uncomment this function:
# def latest_roll_for(team_name):
#     t = team_long[team_long["team"] == team_name]
#     if t.empty:
#         raise ValueError(f"No history for team {team_name}")
#     return t.sort_values("date").iloc[-1:]

# use a future date so rolling stats represent "up to last known match"
example = pd.DataFrame({
    "date": [df["date"].max() + pd.Timedelta(days=1)],
    "home_team": [home_name],
    "away_team": [away_name]
})

def latest_roll_for(team_name: str):
    t = team_long[team_long["team"] == team_name]
    if t.empty:
        raise ValueError(f"No history for team {team_name}")
    return t.sort_values("date").iloc[-1:]


home_latest = latest_roll_for(home_name).add_prefix("home_")
away_latest = latest_roll_for(away_name).add_prefix("away_")

row = example.merge(home_latest, left_on="home_team", right_on="home_team").merge(
    away_latest, left_on="away_team", right_on="away_team"
)

X_example = row[cat_features + num_features]
proba = pipe.predict_proba(X_example)[0]

print("\nExample fixture:", home_name, "vs", away_name, "on", example.loc[0, "date"].date())
print("Classes:", pipe.classes_)
print("Probabilities (order corresponds to Classes):", proba)