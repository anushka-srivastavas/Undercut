# 🏎️ Undercut
### Monte Carlo F1 Race Strategy Simulator

Undercut is an end to end simulation system that recommends optimal pit-stop strategies for Formula 1 races by combining tire degradation modeling, safety car probability estimation, and Monte Carlo simulation on real historical race data.

Built at the intersection of Statistical Modeling, Simulation, and Motorsport, the platform lets users pick a circuit, choose a tire strategy, and compare candidate pit laps to see which one produces the best expected outcome, accounting for the real chance a safety car makes a given pit stop nearly free.

## 🚀 Live Demo
Try it here: https://undercut.streamlit.app

## ✨ Features

- Fits per circuit, per compound tire degradation curves using fixed effects regression on real FastF1 lap data.
- Models circuit specific safety car probability, including when in the race it tends to strike, not just whether it does.
- Runs thousands of Monte Carlo race simulations per candidate pit lap to compare expected outcomes under uncertainty.
- Visualizes strategy comparisons, including how often a safety car would make a given pit lap effectively free.
- Accessible through an interactive Streamlit web application.
- Lets users configure circuit, tire compounds, race length, and candidate pit laps in real time.

## 🏁 Circuits Modeled

- Qatar (Losail) — high tire degradation, abrasive surface
- Monza — lowest-degradation outlier on the calendar
- Brazil (Interlagos) — altitude and variable weather conditions
- Singapore — street circuit, historically high safety car rate

## 🛠 Tech Stack

**Data & Modeling**
- Python
- pandas / NumPy
- scikit-learn (fixed-effects regression)

**Simulation**
- Monte Carlo methods
- Custom safety car hazard modeling

**Frontend & Deployment**
- Streamlit
- Plotly
- Streamlit Cloud

**Data Source**
- FastF1 (official F1 timing API wrapper)
- 4 circuits, multiple seasons, ~15,000 real race laps

## 🧠 System Architecture

```
FastF1 API
      ↓
Race Lap Data (cached)
      ↓
Tire Degradation Model  +  Safety Car Model
      ↓
Monte Carlo Strategy Simulator
      ↓
Streamlit Interactive App
```

## 📊 Results

- Tire degradation model MAE improved from ~0.7–1.4s to ~0.35–0.7s after correcting for fuel burn-off and driver skill confounds via fixed-effects regression
- Safety car timing modeled per-circuit at decile resolution, correctly reproducing known patterns (e.g. Qatar's SC risk clustering at race start and end, with a quiet middle)
- Simulator verified to correctly link safety car timing to pit-stop cost — candidate pit laps inside historical SC windows show measurably higher "SC helped" rates in simulation

## 📂 Project Structure

```
undercut/
├── data/
│   ├── fetch_data.py
│   └── processed/
├── modeling/
│   ├── tire_degradation.py
│   ├── safety_car_model.py
│   └── monte_carlo.py
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

## 💡 Motivation

As an F1 fan, I wanted to build something closer to what real race strategists actually do at the pit wall, rather than another race winner predictor. Undercut combines statistical modeling and simulation to turn raw lap timing data into an actual strategic decision tool and along the way, debugging a genuinely wrong tire degradation model taught me more about regression confounds and fixed effects than any tutorial project could have.
