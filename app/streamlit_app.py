"""
streamlit_app.py
Interactive UI for the F1 race strategy simulator. Lets you pick a circuit
and compound pair, choose candidate pit laps, and see simulated outcomes
compared side by side — including how often a safety car would have made
a given pit lap effectively free.
"""

import sys
from pathlib import Path

# Make the modeling/ folder importable from here.
sys.path.insert(0, str(Path(__file__).parent.parent / "modeling"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from monte_carlo import compare_pit_strategies

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

st.set_page_config(page_title="F1 Strategy Simulator", layout="wide")
st.title("F1 Race Strategy Simulator")
st.caption(
    "Monte Carlo pit-strategy comparison using tire degradation models "
    "fitted on real FastF1 lap data, plus circuit-specific safety car timing."
)

# --- Sidebar controls ---------------------------------------------------
degradation_df = pd.read_parquet(PROCESSED_DIR / "degradation_curves.parquet")
available_circuits = sorted(degradation_df["circuit"].unique())

st.sidebar.header("Race setup")
circuit = st.sidebar.selectbox("Circuit", available_circuits)

available_compounds = sorted(
    degradation_df.loc[degradation_df["circuit"] == circuit, "compound"].unique()
)
compound_stint1 = st.sidebar.selectbox("Stint 1 compound", available_compounds, index=0)
compound_stint2 = st.sidebar.selectbox(
    "Stint 2 compound", available_compounds,
    index=min(1, len(available_compounds) - 1),
)

total_laps = st.sidebar.slider("Total race laps", min_value=40, max_value=70, value=57)

st.sidebar.subheader("Candidate pit laps to compare")
pit_lap_input = st.sidebar.text_input(
    "Comma-separated lap numbers", value="10,20,30,40,50"
)
n_simulations = st.sidebar.slider(
    "Simulations per strategy", min_value=500, max_value=10000, value=3000, step=500
)

run_button = st.sidebar.button("Run simulation", type="primary")

# --- Main panel -----------------------------------------------------------
if run_button:
    try:
        candidate_pit_laps = [int(x.strip()) for x in pit_lap_input.split(",") if x.strip()]
    except ValueError:
        st.error("Pit laps must be a comma-separated list of whole numbers, e.g. 10,20,30")
        st.stop()

    invalid = [p for p in candidate_pit_laps if p < 1 or p >= total_laps]
    if invalid:
        st.error(f"These pit laps are outside the race length (1 to {total_laps - 1}): {invalid}")
        st.stop()

    with st.spinner(f"Running {len(candidate_pit_laps)} strategies x {n_simulations} simulations each..."):
        results = compare_pit_strategies(
            circuit=circuit,
            compound_stint1=compound_stint1,
            compound_stint2=compound_stint2,
            total_laps=total_laps,
            candidate_pit_laps=candidate_pit_laps,
            n_simulations=n_simulations,
        )

    results_df = pd.DataFrame([vars(r) for r in results]).sort_values("mean_total_time")
    best = results_df.iloc[0]

    st.success(
        f"Best expected strategy: pit on lap {int(best['pit_lap'])} "
        f"({compound_stint1} to {compound_stint2}) — "
        f"mean relative time {best['mean_total_time']:.2f}s"
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=results_df["pit_lap"].astype(str),
            y=results_df["mean_total_time"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=results_df["p90_total_time"] - results_df["mean_total_time"],
                arrayminus=results_df["mean_total_time"] - results_df["p10_total_time"],
            ),
            marker_color="crimson",
            name="Mean relative time",
        ))
        fig.update_layout(
            title="Expected outcome by pit lap (lower is better; bars show P10-P90 range)",
            xaxis_title="Pit lap",
            yaxis_title="Relative race time (s)",
            height=450,
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Safety car influence")
        sc_fig = go.Figure()
        sc_fig.add_trace(go.Bar(
            x=results_df["pit_lap"].astype(str),
            y=results_df["sc_helped_fraction"] * 100,
            marker_color="goldenrod",
        ))
        sc_fig.update_layout(
            title="% of simulations where SC made this pit lap cheap",
            xaxis_title="Pit lap",
            yaxis_title="SC-helped %",
            height=450,
        )
        st.plotly_chart(sc_fig, use_container_width=True)

    st.subheader("Full results")
    st.dataframe(
        results_df.rename(columns={
            "pit_lap": "Pit Lap", "mean_total_time": "Mean Time (s)",
            "std_total_time": "Std Dev (s)", "p10_total_time": "P10 (s)",
            "p90_total_time": "P90 (s)", "sc_helped_fraction": "SC Helped %",
        }).assign(**{"SC Helped %": lambda d: (d["SC Helped %"] * 100).round(1)}),
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info("Set up a race scenario in the sidebar and click **Run simulation**.")
