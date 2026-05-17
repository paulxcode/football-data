# European Football Analysis (2008-2016)

Exploratory data analysis of 25,979 matches across 11 European leagues, with interactive visualizations and a match prediction model. All data is precomputed and served as a static site.

**Live demo:** [.vercel.app]

---

## Sections

### Goal Distribution

Most matches are low-scoring (0-3 total goals). The most common scoreline is 1-1.

-here screenshot of goal distribution histogram showing home team, away team, and total goals overlayed-

### Home Advantage

Home teams win 45.87% of matches, a consistent pattern across every league.

-here screenshot of pie chart and bar chart showing home win / draw / away win distribution-

### Season Trends

Goal scoring and home advantage stayed remarkably stable from 2008 to 2016.

-here screenshot of two line charts showing avg goals per season and home win percentage-

### League Comparison

The Netherlands Eredivisie leads with 3.08 goals per match. France Ligue 1 is the most defensive.

-here screenshot of horizontal bar chart of leagues by avg goals and scatter plot-

### Player Analysis

Top 20 players by goals and assists, parsed from XML event data across 14K matches. Messi (384 G+A) edges Ronaldo (355 G+A).

-here screenshot of top 20 table and Messi vs Ronaldo season comparison line chart-

### Match Prediction Model

A Random Forest classifier using B365 betting odds achieves 55.2% accuracy. Home Win is easiest to predict (F1=0.68), Draw is hardest (F1=0.09).

-here screenshot of confusion matrix heatmap and feature importance bar chart-

---

## Project Structure

```
football-analysis-site/
  index.html         single page site (dark theme, Plotly charts)
  style.css          all styling
  data.js            precomputed data payload (198KB)
  export_data.py     Python pipeline: SQLite to data.js
  vercel.json        static site config for Vercel
  README.md          this file
```

---

## How It Works

### Data Pipeline

`export_data.py` connects to a 313MB SQLite database, runs queries, computes statistics, generates Plotly charts, trains a Random Forest model, and serializes everything into `data.js`.

Key queries:

```python
matches_query = '''
    SELECT M.id, M.season, M.date, C.name AS country, L.name AS league,
           M.home_team_goal, M.away_team_goal,
           M.B365H, M.B365D, M.B365A
    FROM Match M
    JOIN Country C ON M.country_id = C.id
    JOIN League L ON M.league_id = L.id
'''
```

### Goal and Assist Parsing

Goal events are stored as XML in the `Match.goal` column. The parser extracts scorers and assisters:

```python
root = ET.fromstring(f"<root>{row['goal']}</root>")
for value in root.findall('.//value'):
    scorer = value.find('player1')
    assister = value.find('player2')
```

14,217 of 25,979 matches have goal event data. This feeds both the top 20 table and the Messi vs Ronaldo comparison.

### Prediction Model

A Random Forest classifier with 200 trees and max depth of 10, trained on B365 betting odds:

```python
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X, y)
```

The model predicts three classes: Home Win, Draw, Away Win. Betting odds are the only features, so the 55.2% accuracy reflects how well market prices encode match outcomes.

### Frontend

`index.html` loads `data.js` (which sets `window.SITE_DATA`) and renders charts via Plotly.js. No build step, no server, no database at runtime.

```javascript
const DATA = window.SITE_DATA;
const CHARTS = DATA.charts;
function chart(id, key) {
  const fig = CHARTS[key];
  Plotly.newPlot(id, fig.data, fig.layout, { responsive: true, displayModeBar: false });
}
```

---

## Setup

### Prerequisites

- Python 3.8+
- [European Soccer Database](https://huggingface.co/datasets/julien-c/kaggle-hugomathien-soccer) SQLite file at `../footballData/database.sqlite`

### Install and Run

```bash
cd ../footballData
python -m venv venv
venv\Scripts\pip install pandas numpy plotly scikit-learn
```

Generate the data payload:

```bash
cd ../football-analysis-site
../footballData/venv/Scripts/python export_data.py
```

Open `index.html` in a browser (works directly from the filesystem).

### Deploy to Vercel

```bash
npx vercel --prod
```

---

## Data Source

[European Soccer Database](https://huggingface.co/datasets/julien-c/kaggle-hugomathien-soccer) by Hugo Mathien, hosted on Hugging Face. Contains match results, player attributes, team info, and betting odds from 11 European leagues across 8 seasons.

---

## Built With

- Python, Pandas, NumPy for data processing
- Scikit-learn for the Random Forest model
- Plotly for interactive visualizations
- Vanilla HTML, CSS, JavaScript for the frontend
