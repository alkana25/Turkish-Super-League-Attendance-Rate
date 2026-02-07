#AHMET UTKU ALKAN
#040220116
#ELECTRONICS AND COMMUNICATION ENGINEERING
#ANN FINAL PROJECT
#ONUR ERGEN
#2025
import pandas as pd
import numpy as np
import unicodedata
import matplotlib.pyplot as plt


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor


FILE_PATH = r"C:\Users\lenovo\Desktop\ANN_PROJECT\superlig_attendance.xlsx"
#4 BIG TEAMS WHICH HAVE HIGHER SPECTATORS
BIG_FOUR = {"Galatasaray", "Fenerbahce", "Besiktas", "Trabzonspor"}
#MACHINE LEARNING AND MATHS EFFECT
RULE_ATTRIBUTION = 0.65
ML_ATTRIBUTION = 0.35

#LOADING DATA FROM EXCEL FILE 
def data_load(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    required = {
        "match_id", "home_team", "away_team",
        "attendance", "capacity", "home_rank", "away_rank"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing Columns of Excel: {missing}")

    if "derby" not in df.columns:
        df["derby"] = 0

    df["home_rank"] = df["home_rank"].astype(int)
    df["away_rank"] = df["away_rank"].astype(int)

    df["occupancy"] = (
        df["attendance"] / df["capacity"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)

    return df

#PRINT ALL TEAMS TO PROVIDE USERS TO SELECT HOME AND AWAYS TEAMS
#KOCAELISPOR AND GENCLERBIRLIGI ARE INCLUDED IN 2025/2026 SEASON BUT THEY ARE NOT ON THE EXCEL LIST BECAUSE THEY WERE NOT IN TURKISH SUPER LEAGUE FOR 3 YEARS
def all_teams(df: pd.DataFrame):
    teams = sorted(set(df["home_team"]).union(set(df["away_team"])))
    print("\n--- TEAMS ---")
    for t in teams:
        print("-", t)
    return teams

#IF THE USER ENTER TURKISH LETTERS SYSTEM WILL UNDERSTAND
def correction_for_team_names(name: str, teams: list[str]) -> str:
    def norm(s):
        s = s.lower()
        s = s.replace("ş", "s").replace("ç", "c").replace("ğ", "g") \
             .replace("ö", "o").replace("ü", "u").replace("ı", "i")
        s = unicodedata.normalize("NFKD", s)
        return s

    n = norm(name)
    for t in teams:
        if norm(t) == n:
            return t
    return name

#RANKS ARE DEFINED BECAUSE THERE IS NOT SAME AMOUNT OF TEAMS EACH YEAR
#FOREXAMPLE: 21 TEAMS FOR 23/24 SEASON AND 19 TEAMS FOR 24/25 SEASON
def rank_explanation():
    print("\n--- RANKS ---")
    print("Rank 1: Championship Tier")
    print("Rank 2: Challenge for European Leagues")
    print("Rank 3: Middle-Higher")
    print("Rank 4: Middle - Lower")
    print("Rank 5: Relegation battle")

#WHEN THE USER ENTER THE HOME AND AWAY TEAMS SYSTEM WILL CATH THE MATCH IS DERBY OR NOT 
def find_derby_matches(df, home, away):
    m = df[(df["home_team"] == home) & (df["away_team"] == away)]
    if len(m) == 0:
        return 0
    return int(m["derby"].max())

#BIG FOUR FANS EFFECT
def Big_four_derby_attendance(df):
    ts = df[
        (df["home_team"] == "Trabzonspor") & 
        # Last three seasons Galatasaray and Fenerbahce are always in championship race and Besiktas is always in European League race. So big team behavior is examined for Trabzonspor
        (df["away_team"].isin(BIG_FOUR)) &
        (df["derby"] == 1)
    ]

    if len(ts) < 5:
        return 0.7

    ch = abs(np.corrcoef(ts["home_rank"], ts["occupancy"])[0, 1])
    ca = abs(np.corrcoef(ts["away_rank"], ts["occupancy"])[0, 1])

    if np.isnan(ch) or np.isnan(ca):
        return 0.7

    return float(np.clip(ch / (ch + ca + 1e-6), 0.6, 0.85))

#AI TRAINING
def train_of_ai(df):
    X = df[["home_team", "away_team", "home_rank", "away_rank", "derby", "capacity"]]
    y = df["occupancy"]

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["home_team", "away_team"]),
            ("num", "passthrough",
             ["home_rank", "away_rank", "derby", "capacity"]),
        ]
    )

    model = Pipeline(
        [
            ("pre", pre),
            ("gb", GradientBoostingRegressor(random_state=42)),
        ]
    )

    model.fit(X, y)
    return model

#DERIVED NEW PROFILES FROM EXCEL FILE
def derived_profiles(df):
    league_avg = float(df["occupancy"].mean())

    home_baseline = (
        df.groupby("home_team")
          .agg(base_occ=("occupancy", "mean"),
               mean_cap=("capacity", "mean"))
          .reset_index()
    )
    home_baseline_lu = {
        r.home_team: (float(r.base_occ), float(r.mean_cap))
        for r in home_baseline.itertuples()
    }

    exact = (
        df.groupby(["home_team", "away_team", "home_rank", "away_rank"])
          .agg(n=("match_id", "count"),
               occ=("occupancy", "mean"))
          .reset_index()
    )
    exact_lu = {
        (r.home_team, r.away_team, int(r.home_rank), int(r.away_rank)):
        (int(r.n), float(r.occ))
        for r in exact.itertuples()
    }

    pair_hr = (
        df.groupby(["home_team", "away_team", "home_rank"])
          .agg(n=("match_id", "count"),
               occ=("occupancy", "mean"))
          .reset_index()
    )
    pair_hr_lu = {
        (r.home_team, r.away_team, int(r.home_rank)):
        (int(r.n), float(r.occ))
        for r in pair_hr.itertuples()
    }

    pair_hstrcl = (
        df.groupby(["home_team", "away_team"])
          .agg(n=("match_id", "count"),
               occ=("occupancy", "mean"))
          .reset_index()
    )
    pair_hstrcl_lu = {
        (r.home_team, r.away_team):
        (int(r.n), float(r.occ))
        for r in pair_hstrcl.itertuples()
    }

    home_hr = (
        df.groupby(["home_team", "home_rank"])
          .agg(n=("match_id", "count"),
               occ=("occupancy", "mean"))
          .reset_index()
    )
    home_hr_lu = {
        (r.home_team, int(r.home_rank)):
        (int(r.n), float(r.occ))
        for r in home_hr.itertuples()
    }

    home_rank_effect = (df.groupby("home_rank")["occupancy"].mean() / league_avg).to_dict()
    away_rank_effect = (df.groupby("away_rank")["occupancy"].mean() / league_avg).to_dict()

    return {
        "league_avg": league_avg,
        "home_baseline_lu": home_baseline_lu,
        "exact_lu": exact_lu,
        "pair_hr_lu": pair_hr_lu,
        "pair_hstrcl_lu": pair_hstrcl_lu,
        "home_hr_lu": home_hr_lu,
        "home_rank_effect": home_rank_effect,
        "away_rank_effect": away_rank_effect,
        "alpha": Big_four_derby_attendance(df),
    }

#PREDICTION OF OCCUPANCY
def predict_occupancy(home, away, hr, ar, derby, lu):
    exact_lu = lu["exact_lu"]
    pair_hr_lu = lu["pair_hr_lu"]
    pair_hstrcl_lu = lu["pair_hstrcl_lu"]
    home_hr_lu = lu["home_hr_lu"]
    home_baseline_lu = lu["home_baseline_lu"]
    league_avg = lu["league_avg"]
    home_rank_effect = lu["home_rank_effect"]
    away_rank_effect = lu["away_rank_effect"]
    alpha = lu["alpha"]

    if home in home_baseline_lu:
        base_occ, cap = home_baseline_lu[home]
    else:
        base_occ, cap = league_avg, 20000

    k_exact = (home, away, hr, ar)
    if k_exact in exact_lu:
        n, occ = exact_lu[k_exact]
        return occ, cap, f"avg_exct(n={n})"

    k_pair_hr = (home, away, hr)
    if k_pair_hr in pair_hr_lu:
        n, occ = pair_hr_lu[k_pair_hr]
        return occ, cap, f"home_rank_avg_pair (n={n})"

    if derby == 1 and home in BIG_FOUR and away in BIG_FOUR:
        h = home_rank_effect.get(hr, 1.0)
        a = min(away_rank_effect.get(ar, 1.0), 1.0)
        return base_occ * (alpha * h + (1 - alpha) * a), cap, "transfer_big_derby"

    k_pair = (home, away)
    if k_pair in pair_hstrcl_lu:
        n, occ = pair_hstrcl_lu[k_pair]
        return occ, cap, f"avg_pair (n={n})"

    k_hr = (home, hr)
    if k_hr in home_hr_lu:
        n, occ = home_hr_lu[k_hr]
        return occ, cap, f"home_rank_behavior (n={n})"

    h = home_rank_effect.get(hr, 1.0)
    a = min(away_rank_effect.get(ar, 1.0), 1.0)
    return base_occ * (0.75 * h + 0.25 * a), cap, "fallback_league"

#PREDICTION
def predict_attendance(home, away, hr, ar, df, ai, lu):
    derby = find_derby_matches(df, home, away)
    rule_occ, capacity, source = predict_occupancy(home, away, hr, ar, derby, lu)

    X = pd.DataFrame([{
        "home_team": home,
        "away_team": away,
        "home_rank": hr,
        "away_rank": ar,
        "derby": derby,
        "capacity": capacity,
    }])

    ml_occ = float(np.clip(ai.predict(X)[0], 0, 1))
    final_occ = RULE_ATTRIBUTION * rule_occ + ML_ATTRIBUTION * ml_occ
    final_occ = float(np.clip(final_occ, 0, 1))

    return {
        "attendance": int(round(final_occ * capacity)),
        "occupancy": round(final_occ, 3),
        "rule": round(rule_occ, 3),
        "ai": round(ml_occ, 3),
        "capacity": int(capacity),
        "derby": derby,
        "source": source,
    }

if __name__ == "__main__":
    dataset = data_load(FILE_PATH)

    WHOLE_TEAMS = all_teams(dataset)
    rank_explanation()

    lookups = derived_profiles(dataset)
    ai_model = train_of_ai(dataset)

    selection = input(
        "\nSelect mode:\n"
        "- '1': Attendance Prediction\n"
        "- '2': Plots\n"
        "> "
    ).strip().lower()
#PREDICTION ATTENDANCE RATE
    if selection == "1":
        print("\n=== Turkish Super League Attendance Rate Estimation ===")
        while True:
            input_home = input("\nHome team (press q for exit ):\n> ").strip()
            if input_home.lower() == "q":
                break

            input_away = input("Away team:\n> ").strip()

            home = correction_for_team_names(input_home, WHOLE_TEAMS)
            away = correction_for_team_names(input_away, WHOLE_TEAMS)

            input_home_rank = int(input("Home rank (1-5): "))
            input_away_rank = int(input("Away rank (1-5): "))

            r = predict_attendance(
                home,
                away,
                input_home_rank,
                input_away_rank,
                dataset,
                ai_model,
                lookups
            )

            print("\n--- RESULTS ---")
            print("Estimated attendance :", r["attendance"])
            print("Capacity       :", r["capacity"])
            print("Occupancy rate :", r["occupancy"])
            print("Derby          :", r["derby"])

if selection == "2":

#ACTUAL vs PREDICTED PLOT
    actual = []
    predicted = []

    for r in dataset.itertuples():
        out = predict_attendance(
            r.home_team,
            r.away_team,
            int(r.home_rank),
            int(r.away_rank),
            dataset,
            ai_model,
            lookups
        )
        actual.append(r.occupancy)
        predicted.append(out["occupancy"])

    plt.figure()
    plt.scatter(actual, predicted)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("Actual Occupancy")
    plt.ylabel("Predicted Occupancy")
    plt.title("Actual vs Predicted Occupancy")
    plt.show()

#RANK EFFECT PLOT
    home_rank_mean = dataset.groupby("home_rank")["occupancy"].mean()
    away_rank_mean = dataset.groupby("away_rank")["occupancy"].mean()

    plt.figure()
    plt.plot(home_rank_mean.index, home_rank_mean.values, marker="o", label="Home Rank Effect")
    plt.plot(away_rank_mean.index, away_rank_mean.values, marker="o", label="Away Rank Effect")
    plt.xlabel("Rank")
    plt.ylabel("Average Occupancy")
    plt.title("Rank Effect on Stadium Occupancy")
    plt.legend()
    plt.show()

#DERBY PLOT
    derby_mean = dataset.groupby("derby")["occupancy"].mean()

    labels_derby = ["Derby", "Non-Derby"]
    values_derby = [derby_mean.get(1, 0), derby_mean.get(0, 0)]

    plt.figure()
    plt.bar(labels_derby, values_derby)
    plt.ylabel("Average Occupancy")
    plt.title("Derby / Non-Derby Effect on Stadium Occupancy")
    plt.show()

    exit()



