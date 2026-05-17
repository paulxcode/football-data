import sqlite3
import json
import os
import warnings
import xml.etree.ElementTree as ET
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from plotly.utils import PlotlyJSONEncoder

warnings.filterwarnings("ignore")

DB_PATH = os.path.join(os.path.dirname(__file__), "database.sqlite")
DATA_JS_PATH = os.path.join(os.path.dirname(__file__), "data.js")

conn = sqlite3.connect(DB_PATH)

matches_query = """
    SELECT
        M.id,
        M.season,
        M.date,
        C.name AS country,
        L.name AS league,
        M.home_team_goal,
        M.away_team_goal,
        M.B365H, M.B365D, M.B365A
    FROM Match M
    JOIN Country C ON M.country_id = C.id
    JOIN League L ON M.league_id = L.id
"""
df = pd.read_sql(matches_query, conn)
df["date"] = pd.to_datetime(df["date"])
df["total_goals"] = df["home_team_goal"] + df["away_team_goal"]

total_matches = len(df)
total_players = pd.read_sql("SELECT COUNT(*) as c FROM Player", conn).iloc[0]["c"]
total_teams = pd.read_sql("SELECT COUNT(*) as c FROM Team", conn).iloc[0]["c"]
total_seasons = df["season"].nunique()
date_min = df["date"].min().strftime("%Y-%m-%d")
date_max = df["date"].max().strftime("%Y-%m-%d")
total_goals_scored = int(df["total_goals"].sum())

home_wins = df[df["home_team_goal"] > df["away_team_goal"]].shape[0]
draws = df[df["home_team_goal"] == df["away_team_goal"]].shape[0]
away_wins = df[df["home_team_goal"] < df["away_team_goal"]].shape[0]

home_win_pct = round(home_wins / total_matches * 100, 2)
draw_pct = round(draws / total_matches * 100, 2)
away_win_pct = round(away_wins / total_matches * 100, 2)
avg_goals = round(df["total_goals"].mean(), 2)

most_common_score = (
    df.groupby(["home_team_goal", "away_team_goal"])
    .size()
    .idxmax()
)
most_common_score = f"{most_common_score[0]}-{most_common_score[1]}"

league_avg_goals = (
    df.groupby("league")["total_goals"].mean().sort_values(ascending=False)
)
highest_scoring_league = league_avg_goals.index[0]
highest_scoring_avg = round(league_avg_goals.iloc[0], 2)

# --- PLOTLY CHARTS ---
template = "plotly_dark"
font_cfg = dict(color="#e2e8f0")
paper_bg = "rgba(0,0,0,0)"
px_kwargs = dict(template=template)

def apply_layout(fig):
    fig.update_layout(paper_bgcolor=paper_bg, font=font_cfg)
    return fig

charts = {}

# Goal distribution (3 histograms)
goal_cols = df[["home_team_goal", "away_team_goal", "total_goals"]].rename(columns={
    "home_team_goal": "Home Team",
    "away_team_goal": "Away Team",
    "total_goals": "Total",
})
fig_goal_dist = px.histogram(
    goal_cols,
    labels={"value": "Goals", "variable": "Type"},
    title="Goal Distribution",
    nbins=20,
    barmode="overlay",
    opacity=0.7,
    **px_kwargs,
)
apply_layout(fig_goal_dist)
fig_goal_dist.update_layout(
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
)
charts["goal_distribution"] = fig_goal_dist.to_dict()

# Home advantage pie chart
labels_pie = ["Home Win", "Draw", "Away Win"]
values_pie = [home_wins, draws, away_wins]
fig_home_adv_pie = go.Figure(
    data=[go.Pie(labels=labels_pie, values=values_pie, hole=0.4)],
    layout=go.Layout(
        title="Match Outcomes Distribution",
        template=template,
        font=font_cfg,
    ),
)
apply_layout(fig_home_adv_pie)
charts["home_advantage_pie"] = fig_home_adv_pie.to_dict()

# Home advantage bar chart
fig_home_adv_bar = go.Figure(
    data=[
        go.Bar(
            x=labels_pie,
            y=[home_win_pct, draw_pct, away_win_pct],
            text=[f"{v}%" for v in [home_win_pct, draw_pct, away_win_pct]],
            textposition="outside",
        )
    ],
    layout=go.Layout(
        title="Match Outcomes Percentage",
        yaxis=dict(title="Percentage (%)"),
        template=template,
        font=font_cfg,
    ),
)
apply_layout(fig_home_adv_bar)
charts["home_advantage_bar"] = fig_home_adv_bar.to_dict()

# Season trends: avg goals over time
season_avg_goals = df.groupby("season")["total_goals"].mean().reset_index()
season_avg_goals.columns = ["season", "avg_goals"]
fig_season_goals = go.Figure(
    data=[go.Scatter(x=season_avg_goals["season"], y=season_avg_goals["avg_goals"], mode="lines+markers")],
    layout=go.Layout(
        title="Average Goals Per Match by Season",
        xaxis=dict(title="Season"),
        yaxis=dict(title="Avg Goals"),
        template=template,
        font=font_cfg,
    ),
)
apply_layout(fig_season_goals)
charts["season_trend_goals"] = fig_season_goals.to_dict()

# Season trends: home advantage over time
season_outcomes = (
    df.groupby("season")
    .apply(
        lambda x: pd.Series(
            {
                "home_win_pct": (x["home_team_goal"] > x["away_team_goal"]).mean()
                * 100
            }
        )
    )
    .reset_index()
)
fig_season_home_adv = go.Figure(
    data=[go.Scatter(x=season_outcomes["season"], y=season_outcomes["home_win_pct"], mode="lines+markers")],
    layout=go.Layout(
        title="Home Win Percentage by Season",
        xaxis=dict(title="Season"),
        yaxis=dict(title="Home Win %"),
        template=template,
        font=font_cfg,
    ),
)
apply_layout(fig_season_home_adv)
charts["season_trend_home_adv"] = fig_season_home_adv.to_dict()

# League comparison: horizontal bar chart
league_avg = df.groupby("league")["total_goals"].mean().sort_values().reset_index()
league_avg.columns = ["league", "avg_goals"]
fig_league_bar = go.Figure(data=[
    go.Bar(x=league_avg["avg_goals"], y=league_avg["league"], orientation="h",
           hovertemplate="Avg %{x:.2f}<extra></extra>")
])
fig_league_bar.update_layout(
    title="Average Goals per Match by League",
    xaxis=dict(title="Avg Goals"),
    template=template,
    font=font_cfg,
)
apply_layout(fig_league_bar)
charts["league_comparison_bar"] = fig_league_bar.to_dict()

# League comparison: scatter plot with visible labels
league_stats = (
    df.groupby("league")
    .agg(avg_goals=("total_goals", "mean"), num_matches=("id", "count"))
    .reset_index()
)
fig_league_scatter = go.Figure()
league_stats["short_label"] = league_stats["league"].str.split(" ", n=1).str[1]
fig_league_scatter.add_trace(go.Scatter(
    x=league_stats["avg_goals"],
    y=league_stats["num_matches"],
    mode="markers+text",
    text=league_stats["short_label"],
    textposition="top center",
    marker=dict(size=10, color="#7c3aed", line=dict(color="#e2e8f0", width=1)),
    hovertemplate="<b>%{text}</b><br>Avg Goals: %{x:.2f}<br>Matches: %{y}<extra></extra>",
))
fig_league_scatter.update_layout(
    title="League Comparison: Goals vs Matches",
    xaxis=dict(title="Avg Goals"),
    yaxis=dict(title="Number of Matches"),
    template=template,
    font=font_cfg,
)
apply_layout(fig_league_scatter)
charts["league_comparison_scatter"] = fig_league_scatter.to_dict()

# Top 20 players bar chart
print("Parsing goal/assist data from match events...")
matches_with_goals = pd.read_sql("SELECT id, goal FROM Match WHERE goal IS NOT NULL", conn)

match_season = dict(zip(df["id"], df["season"]))

goals_per_player = {}
assists_per_player = {}
player_season = {}

for _, row in matches_with_goals.iterrows():
    match_id = row["id"]
    season = match_season.get(match_id)
    try:
        root = ET.fromstring(f"<root>{row['goal']}</root>")
        for value in root.findall(".//value"):
            scorer = value.find("player1")
            assister = value.find("player2")
            if scorer is not None and scorer.text:
                pid = int(scorer.text)
                goals_per_player[pid] = goals_per_player.get(pid, 0) + 1
                if season:
                    player_season.setdefault(pid, {}).setdefault(season, {"g": 0, "a": 0})["g"] += 1
            if assister is not None and assister.text:
                pid = int(assister.text)
                if pid >= 0:
                    assists_per_player[pid] = assists_per_player.get(pid, 0) + 1
                    if season:
                        player_season.setdefault(pid, {}).setdefault(season, {"g": 0, "a": 0})["a"] += 1
    except ET.ParseError:
        continue

goal_df = pd.DataFrame(list(goals_per_player.items()), columns=["player_api_id", "goals"])
assist_df = pd.DataFrame(list(assists_per_player.items()), columns=["player_api_id", "assists"])
stats_df = goal_df.merge(assist_df, on="player_api_id", how="outer").fillna(0)
stats_df["total"] = stats_df["goals"] + stats_df["assists"]
stats_df = stats_df.astype({"goals": int, "assists": int, "total": int})

players_names = pd.read_sql("SELECT player_api_id, player_name FROM Player", conn)
stats_df = stats_df.merge(players_names, on="player_api_id").sort_values("total", ascending=False).head(20)

row_colors = ["#0f172a", "#1a2332"]

fig_top_players = go.Figure(data=[go.Table(
    header=dict(
        values=["Player", "Goals", "Assists", "G+A"],
        align=["left", "center", "center", "center"],
        font=dict(size=13, color="#e2e8f0", family="Arial"),
        fill_color="#1e293b",
        line_color="#334155",
        height=36,
    ),
    cells=dict(
        values=[
            stats_df["player_name"].tolist(),
            stats_df["goals"].apply(lambda v: f"{v:.0f}").tolist(),
            stats_df["assists"].apply(lambda v: f"{v:.0f}").tolist(),
            stats_df["total"].apply(lambda v: f"{v:.0f}").tolist(),
        ],
        align=["left", "center", "center", "center"],
        font=dict(size=12, color="#e2e8f0", family="Arial"),
        fill_color=[[row_colors[i % 2] for i in range(len(stats_df))]],
        line_color="#334155",
        height=34,
    ),
)])
fig_top_players.update_layout(
    title="Top 20 Players by Goals + Assists",
    template=template,
    font=font_cfg,
    height=720,
    margin=dict(l=20, r=20, t=50, b=20),
)
apply_layout(fig_top_players)
charts["top_20_players"] = fig_top_players.to_dict()

# Messi vs Ronaldo: season-by-season comparison
players_df = pd.read_sql("SELECT player_api_id, player_name, birthday FROM Player", conn)
messi_id = players_df[players_df["player_name"].str.contains("Messi", case=False)]["player_api_id"].iloc[0]
ronaldo_id = players_df[players_df["player_name"].str.contains("Ronaldo", case=False)]["player_api_id"].iloc[0]

seasons_sorted = sorted(df["season"].unique())
messi_ga = [player_season.get(messi_id, {}).get(s, {"g": 0, "a": 0}) for s in seasons_sorted]
ronaldo_ga = [player_season.get(ronaldo_id, {}).get(s, {"g": 0, "a": 0}) for s in seasons_sorted]

fig_mr = go.Figure()
fig_mr.add_trace(go.Scatter(
    x=seasons_sorted, y=[d["g"] + d["a"] for d in messi_ga],
    mode="lines+markers",
    name="Messi",
    line=dict(color="#3b82f6", width=3),
    marker=dict(size=8),
    customdata=[[d["g"], d["a"]] for d in messi_ga],
    hovertemplate="<b>Messi</b><br>Season: %{x}<br>Goals: %{customdata[0]}<br>Assists: %{customdata[1]}<br>Total: %{y}<extra></extra>",
))
fig_mr.add_trace(go.Scatter(
    x=seasons_sorted, y=[d["g"] + d["a"] for d in ronaldo_ga],
    mode="lines+markers",
    name="Ronaldo",
    line=dict(color="#ef4444", width=3),
    marker=dict(size=8),
    customdata=[[d["g"], d["a"]] for d in ronaldo_ga],
    hovertemplate="<b>Ronaldo</b><br>Season: %{x}<br>Goals: %{customdata[0]}<br>Assists: %{customdata[1]}<br>Total: %{y}<extra></extra>",
))
fig_mr.update_layout(
    title="Messi vs Ronaldo: Goals + Assists per Season",
    xaxis_title="Season",
    yaxis_title="Goals + Assists",
    template=template,
    font=font_cfg,
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)
apply_layout(fig_mr)
charts["messi_ronaldo_comparison"] = fig_mr.to_dict()

# RandomForest model
print("Training RandomForest model on betting odds...")
odds_df = df[df["B365H"].notna()].copy()
odds_df["target"] = odds_df.apply(
    lambda r: 0 if r["home_team_goal"] > r["away_team_goal"]
    else (1 if r["home_team_goal"] == r["away_team_goal"] else 2),
    axis=1,
)

feature_cols = ["B365H", "B365D", "B365A"]
X = odds_df[feature_cols].values
y = odds_df["target"].values

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X, y)

y_pred = rf.predict(X)
report_dict = classification_report(y, y_pred, target_names=["Home Win", "Draw", "Away Win"], output_dict=True)
cm = confusion_matrix(y, y_pred)

charts["classification_report"] = report_dict

# Confusion matrix heatmap
fig_cm = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=["Home Win (pred)", "Draw (pred)", "Away Win (pred)"],
        y=["Home Win (true)", "Draw (true)", "Away Win (true)"],
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
    ),
    layout=go.Layout(
        title="Confusion Matrix - RandomForest",
        template=template,
        font=font_cfg,
    ),
)
apply_layout(fig_cm)
charts["confusion_matrix_heatmap"] = fig_cm.to_dict()

# Feature importance bar chart
feature_importance = rf.feature_importances_
fi_labels = {"B365H": "Home Win Odds", "B365D": "Draw Odds", "B365A": "Away Win Odds"}
df_fi = pd.DataFrame({"feature": [fi_labels[c] for c in feature_cols], "importance": feature_importance})
fig_fi = px.bar(
    df_fi,
    x="feature",
    y="importance",
    title="Feature Importance (Betting Odds)",
    labels={"importance": "Importance", "feature": "Odds Type"},
    **px_kwargs,
)
apply_layout(fig_fi)
charts["feature_importance"] = fig_fi.to_dict()

# --- BUILD DATA STRUCTURE ---
data = {
    "total_matches": total_matches,
    "total_players": int(total_players),
    "total_teams": int(total_teams),
    "total_seasons": int(total_seasons),
    "date_range": f"{date_min} to {date_max}",
    "home_win_pct": home_win_pct,
    "draw_pct": draw_pct,
    "away_win_pct": away_win_pct,
    "avg_goals": avg_goals,
    "most_common_score": most_common_score,
    "highest_scoring_league": highest_scoring_league,
    "highest_scoring_avg": highest_scoring_avg,
    "total_goals_scored": total_goals_scored,
    "charts": charts,
}

js_content = f"window.SITE_DATA = {json.dumps(data, cls=PlotlyJSONEncoder)};"

os.makedirs(os.path.dirname(DATA_JS_PATH), exist_ok=True)
with open(DATA_JS_PATH, "w", encoding="utf-8") as f:
    f.write(js_content)

file_size = os.path.getsize(DATA_JS_PATH)
print(f"Exported data to {DATA_JS_PATH}")
print(f"File size: {file_size:,} bytes")

conn.close()
