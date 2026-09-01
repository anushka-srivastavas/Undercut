# Undercut

A Monte Carlo simulation tool that recommends optimal pit-stop strategies for
Formula 1 races, using tire degradation and safety car probability models
trained on real historical race data — the same core question F1 race
strategists solve at the pit wall, tackled with real timing data instead of
guesswork.

**Live app:** [](https://undercut.streamlit.app)
**Data source:** [FastF1](https://github.com/theOehrly/Fast-F1)

## Why this project

This project deliberately avoids the most common F1 ML pattern — predicting
race winners, in favor of a strategy optimization problem closer to what
real race engineers solve. Given a tire's current wear and a circuit's
historical safety car pattern, when should a team actually pit? That
question is harder to templatize than a finishing-position classifier, and
it's the actual job strategy engineers do at the pit wall.

## What it does

1. Pulls real lap-by-lap race data for 4 circuits chosen for contrasting tire
   behavior: Qatar (high degradation), Monza (low degradation),
   Brazil (altitude, variable conditions), Singapore (street circuit,
   historically high safety car rate).
2. Fits a tire degradation model per circuit/compound, estimating how
   much lap time is lost per lap of tire wear.
3. Fits a safety car probability model per circuit, estimating not just
   whether a safety car tends to appear, but when in the race — since
   pitting during a safety car is dramatically cheaper (the whole field
   bunches up and slows together).
4. Runs a Monte Carlo simulation: for each candidate pit lap, simulates
   the race thousands of times with randomized safety car timing drawn from
   the historical distribution, and compares expected outcomes.
5. Presents results in an interactive Streamlit app — pick a circuit,
   tire strategy, and race length, and compare candidate pit laps live.

## Architecture
undercut/
├── data/
│ ├── fetch_data.py # pulls and caches race data via FastF1
│ └── processed/ # fitted model outputs (parquet)
├── modeling/
│ ├── tire_degradation.py # per-circuit/compound degradation regression
│ ├── safety_car_model.py # per-circuit safety car timing model
│ └── monte_carlo.py # simulation engine combining both models
├── app/
│ └── streamlit_app.py # interactive UI
└── requirements.txt


## The interesting part: getting the tire degradation model right

The first version of the tire degradation model produced physically
backwards results — some circuits showed tires getting faster as they
aged. The cause was multicollinearity: within a single tire stint,
"how old is this tire" and "how far into the race are we" move together
almost in lockstep, so a naive regression couldn't tell which one was really
driving the lap-time change.

Two fixes were tried:

1. Hardcoding a fixed fuel-burn constant (from known F1 physics, ~0.03s
   per lap per kg of fuel) to remove the fuel effect before fitting tire age.
   This didn't hold up — it assumes every stint burns fuel identically
   regardless of when in the race it starts, which isn't true.
2. Fixed-effects regression — instead of hardcoding anything, each lap's
   time is compared against (a) the field's median pace on that exact lap
   number that race, removing fuel burn-off and track evolution, and (b)
   each driver's own average pace that race, removing raw driver/car skill
   differences. What's left over is a much cleaner tire-wear signal.

This second approach is what's used in the current model, and it noticeably
improved fit quality (mean absolute error dropped from ~0.7–1.4s to
~0.35–0.7s across circuits).

## Known limitations

- Small sample size. Only 3–4 seasons of data per circuit means safety
  car probabilities are rough estimates, not settled statistics — a single
  race going differently could shift a circuit's rate substantially.
- Possible remaining confounding in the tire model. Real teams don't
  choose tire compounds randomly — a HARD-tire stint likely correlates with
  different race conditions or strategic situations than a MEDIUM-tire
  stint, which the current model can't fully separate from genuine
  compound-level degradation differences. Flagged as a direction for future
  work (e.g., controlling for stint start lap, or race position).
- Simplified safety car cost model. Pit-stop time savings under a safety
  car are modeled as a fixed multiplier rather than simulated from the
  underlying field-bunching dynamics.

## Running it locally

```bash
git clone https://github.com/anushka-srivastavas/undercut.git
cd undercut
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Pull fresh data and refit models (optional — pre-fitted models are included)
python data/fetch_data.py
python modeling/tire_degradation.py
python modeling/safety_car_model.py

# Launch the app
streamlit run app/streamlit_app.py
```

## Tech stack

Python, FastF1, pandas, scikit-learn, NumPy, Streamlit, Plotly

## Data source

Race data via [FastF1](https://github.com/theOehrly/Fast-F1), which sources
from the official F1 live timing API and the community-maintained
[Jolpica-F1](https://api.jolpi.ca/ergast/f1) project (the successor to the
now-retired Ergast API).
