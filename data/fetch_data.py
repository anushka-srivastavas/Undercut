"""
fetch_data.py
Pulls race lap data from FastF1 for 4 circuits chosen for contrasting tire
behavior, and caches it as a single parquet file.

Run this locally:  python data/fetch_data.py
"""

from pathlib import Path
import fastf1
import pandas as pd

CACHE_DIR = Path(__file__).parent / "fastf1_cache"
OUTPUT_DIR = Path(__file__).parent / "processed"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))

# circuit key -> FastF1's event name string
CIRCUITS = {
    "qatar": "Qatar",
    "brazil": "Sao Paulo",
    "monza": "Italy",
    "singapore": "Singapore",
}

SEASONS_BY_CIRCUIT = {
    "qatar": [2021, 2023, 2024],
    "brazil": [2021, 2022, 2023, 2024],
    "monza": [2021, 2022, 2023, 2024],
    "singapore": [2022, 2023, 2024],
}


def fetch_race_laps(circuit_key: str, year: int):
    event_name = CIRCUITS[circuit_key]
    try:
        session = fastf1.get_session(year, event_name, "R")
        session.load(laps=True, telemetry=False, weather=True, messages=True)
    except Exception as e:
        print(f"  [skip] {circuit_key} {year}: {e}")
        return None

    laps = session.laps.copy()
    if laps.empty:
        return None

    keep_cols = ["Driver", "DriverNumber", "LapNumber", "LapTime", "Stint",
                 "Compound", "TyreLife", "FreshTyre", "TrackStatus",
                 "PitInTime", "PitOutTime", "IsAccurate"]
    laps = laps[[c for c in keep_cols if c in laps.columns]].copy()

    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    laps["circuit"] = circuit_key
    laps["year"] = year

    total_laps = laps["LapNumber"].max()
    laps["lap_fraction"] = laps["LapNumber"] / total_laps

    return laps


def fetch_all():
    all_laps = []
    for circuit_key, seasons in SEASONS_BY_CIRCUIT.items():
        for year in seasons:
            print(f"Fetching {circuit_key} {year}...")
            df = fetch_race_laps(circuit_key, year)
            if df is not None:
                all_laps.append(df)

    combined = pd.concat(all_laps, ignore_index=True)
    combined.to_parquet(OUTPUT_DIR / "laps_raw.parquet", index=False)
    print(f"\nSaved {len(combined)} laps -> data/processed/laps_raw.parquet")
    return combined


if __name__ == "__main__":
    fetch_all()