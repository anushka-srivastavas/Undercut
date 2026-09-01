"""
monte_carlo.py
Simulates race strategies (when to pit) thousands of times, factoring in
tire degradation and safety car probability, to compare which pit-lap
choice produces the best EXPECTED outcome across many possible races.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path

from tire_degradation import DegradationCurve
from safety_car_model import SafetyCarModel

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

GREEN_PIT_LOSS_SECONDS = 22.0
SC_PIT_LOSS_MULTIPLIER = 0.35


def load_degradation_curves(circuit, compound):
    df = pd.read_parquet(PROCESSED_DIR / "degradation_curves.parquet")
    row = df[(df["circuit"] == circuit) & (df["compound"] == compound)]
    if row.empty:
        raise ValueError(f"No fitted curve for {circuit}/{compound}")
    row = row.iloc[0]
    return DegradationCurve(
        circuit=row["circuit"], compound=row["compound"],
        base_deviation=row["base_deviation"], deg_rate=row["deg_rate"],
        deg_accel=row["deg_accel"], n_laps_fit=int(row["n_laps_fit"]),
        mae_seconds=row["mae_seconds"], r2=row["r2"],
    )


def load_safety_car_model(circuit):
    df = pd.read_parquet(PROCESSED_DIR / "safety_car_models.parquet")
    row = df[df["circuit"] == circuit]
    if row.empty:
        raise ValueError(f"No safety car model for {circuit}")
    row = row.iloc[0]
    hazard = [row[f"hazard_decile_{i}"] for i in range(10)]
    return SafetyCarModel(
        circuit=row["circuit"], race_level_probability=row["race_level_probability"],
        lap_hazard_by_decile=hazard, n_races=int(row["n_races"]),
    )


@dataclass
class StrategyResult:
    pit_lap: int
    mean_total_time: float
    std_total_time: float
    p10_total_time: float
    p90_total_time: float
    sc_helped_fraction: float


def simulate_one_race(pit_lap, total_laps, curve_stint1, curve_stint2, sc_model, rng):
    total_time = 0.0
    sc_lap = sc_model.sample_sc_lap(total_laps, rng)
    sc_helped = False

    for lap in range(1, total_laps + 1):
        if lap <= pit_lap:
            tyre_age = lap
            lap_time = curve_stint1.predict_relative_time(tyre_age)
        else:
            tyre_age = lap - pit_lap
            lap_time = curve_stint2.predict_relative_time(tyre_age)

        lap_time += rng.normal(0, 0.3)
        total_time += lap_time

        if lap == pit_lap:
            if sc_lap is not None and abs(sc_lap - pit_lap) <= 1:
                total_time += GREEN_PIT_LOSS_SECONDS * SC_PIT_LOSS_MULTIPLIER
                sc_helped = True
            else:
                total_time += GREEN_PIT_LOSS_SECONDS

    return total_time, sc_helped


def compare_pit_strategies(circuit, compound_stint1, compound_stint2, total_laps,
                            candidate_pit_laps, n_simulations=5000, seed=42):
    curve1 = load_degradation_curves(circuit, compound_stint1)
    curve2 = load_degradation_curves(circuit, compound_stint2)
    sc_model = load_safety_car_model(circuit)

    rng = np.random.default_rng(seed)
    results = []

    for pit_lap in candidate_pit_laps:
        times = []
        sc_helps = []
        for _ in range(n_simulations):
            t, helped = simulate_one_race(pit_lap, total_laps, curve1, curve2, sc_model, rng)
            times.append(t)
            sc_helps.append(helped)

        times = np.array(times)
        results.append(StrategyResult(
            pit_lap=pit_lap,
            mean_total_time=float(times.mean()),
            std_total_time=float(times.std()),
            p10_total_time=float(np.percentile(times, 10)),
            p90_total_time=float(np.percentile(times, 90)),
            sc_helped_fraction=float(np.mean(sc_helps)),
        ))

    return results


if __name__ == "__main__":
    results = compare_pit_strategies(
        circuit="qatar", compound_stint1="HARD", compound_stint2="MEDIUM",
        total_laps=57, candidate_pit_laps=[15, 20, 25, 30, 35], n_simulations=5000,
    )

    print(f"{'Pit Lap':>8} {'Mean Time':>12} {'Std Dev':>10} {'P10':>10} {'P90':>10} {'SC Helped %':>12}")
    for r in sorted(results, key=lambda x: x.mean_total_time):
        print(f"{r.pit_lap:>8} {r.mean_total_time:>12.2f} {r.std_total_time:>10.2f} "
              f"{r.p10_total_time:>10.2f} {r.p90_total_time:>10.2f} {r.sc_helped_fraction*100:>11.1f}%")
