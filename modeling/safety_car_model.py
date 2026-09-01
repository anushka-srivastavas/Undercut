"""
safety_car_model.py
Estimates, per circuit, how likely a safety car / VSC is to appear, and in
which *decile* of the race it tends to show up (e.g. Singapore's incidents
cluster early; other circuits spike mid-race).

FastF1's TrackStatus codes (per official docs): '1' = green flag,
'4' = safety car, '5' = red flag, '6' = VSC deployed, '7' = VSC ending.
We treat '4', '6', '7' as an active SC/VSC lap.
"""

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
SC_STATUS_CODES = {"4", "6", "7"}


def _is_sc_lap(track_status: str) -> bool:
    if not isinstance(track_status, str):
        return False
    return any(code in track_status for code in SC_STATUS_CODES)


@dataclass
class SafetyCarModel:
    circuit: str
    race_level_probability: float
    lap_hazard_by_decile: list = field(default_factory=lambda: [0.0] * 10)
    n_races: int = 0

    def sample_sc_lap(self, total_laps: int, rng: np.random.Generator):
        """Draw whether/when a safety car appears this simulated race."""
        decile_size = max(1, total_laps // 10)
        for lap in range(1, total_laps + 1):
            decile_idx = min(9, (lap - 1) // decile_size)
            per_lap_hazard = self.lap_hazard_by_decile[decile_idx] / decile_size
            if rng.random() < per_lap_hazard:
                return lap
        return None


def fit_safety_car_models(laps: pd.DataFrame):
    laps = laps.copy()
    laps["is_sc"] = laps["TrackStatus"].apply(_is_sc_lap)
    models = {}

    for circuit, circ_group in laps.groupby("circuit"):
        race_keys = circ_group[["circuit", "year"]].drop_duplicates()
        n_races = len(race_keys)
        races_with_sc = 0
        deployment_deciles = []

        for (circ, yr), race_laps in circ_group.groupby(["circuit", "year"]):
            race_laps = race_laps.sort_values("LapNumber")
            total_laps = race_laps["LapNumber"].max()
            if not total_laps or pd.isna(total_laps):
                continue
            sc_laps = race_laps.loc[race_laps["is_sc"], "LapNumber"]
            if len(sc_laps) > 0:
                races_with_sc += 1
                first_sc_lap = sc_laps.min()
                decile = min(9, int((first_sc_lap - 1) / total_laps * 10))
                deployment_deciles.append(decile)

        race_level_prob = races_with_sc / n_races if n_races else 0.0
        hazard = [0.0] * 10
        for d in deployment_deciles:
            hazard[d] += 1
        if n_races:
            hazard = [h / n_races for h in hazard]

        models[circuit] = SafetyCarModel(
            circuit=circuit, race_level_probability=race_level_prob,
            lap_hazard_by_decile=hazard, n_races=n_races,
        )
        print(f"  [fit] {circuit}: {races_with_sc}/{n_races} races had SC "
              f"({race_level_prob:.0%})")

    return models


if __name__ == "__main__":
    laps = pd.read_parquet(PROCESSED_DIR / "laps_raw.parquet")
    models = fit_safety_car_models(laps)
    rows = []
    for m in models.values():
        row = {"circuit": m.circuit, "race_level_probability": m.race_level_probability,
               "n_races": m.n_races}
        for i, h in enumerate(m.lap_hazard_by_decile):
            row[f"hazard_decile_{i}"] = h
        rows.append(row)
    pd.DataFrame(rows).to_parquet(PROCESSED_DIR / "safety_car_models.parquet", index=False)
    print(f"\nSaved {len(rows)} safety car models.")