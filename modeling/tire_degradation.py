"""
tire_degradation.py
Fits a per-circuit, per-compound quadratic degradation curve.

Two confounds are removed before fitting, using "fixed effects" (subtracting
group averages) rather than hardcoded constants:

1. Field baseline (per circuit/year/lap number): removes fuel burn-off and
   track evolution, which affect every car on track roughly equally.
2. Driver baseline (per circuit/year/driver): removes each driver/car's own
   average pace level, since faster drivers/cars would otherwise look like
   "no degradation" purely because they're fast on every lap regardless of
   tire age.

What's left after both are removed is, as closely as we can get with this
data, each driver's own trend AWAY from their own average pace as their tires
age — which is what tire degradation actually is.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


@dataclass
class DegradationCurve:
    circuit: str
    compound: str
    base_deviation: float
    deg_rate: float
    deg_accel: float
    n_laps_fit: int
    mae_seconds: float
    r2: float

    def predict_relative_time(self, tyre_age: float) -> float:
        return self.base_deviation + self.deg_rate * tyre_age + self.deg_accel * (tyre_age ** 2)


def _clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["PitInTime"].isna() & df["PitOutTime"].isna()]
    if "TrackStatus" in df.columns:
        df = df[df["TrackStatus"] == "1"]
    if "IsAccurate" in df.columns:
        df = df[df["IsAccurate"] == True]  # noqa: E712
    df = df.dropna(subset=["LapTimeSeconds", "TyreLife", "Compound", "LapNumber", "Driver"])

    df["_median"] = df.groupby(["circuit", "year"])["LapTimeSeconds"].transform("median")
    df = df[df["LapTimeSeconds"] < df["_median"] * 1.15]
    return df.drop(columns="_median")


def fit_degradation_curves(laps: pd.DataFrame, min_laps: int = 25):
    clean = _clean_laps(laps)

    # Step 1: remove field-level effects (fuel burn-off, track evolution).
    clean["field_baseline"] = clean.groupby(
        ["circuit", "year", "LapNumber"]
    )["LapTimeSeconds"].transform("median")
    clean["relative_time"] = clean["LapTimeSeconds"] - clean["field_baseline"]

    # Step 2: remove driver/car-level effects (raw pace differences between
    # drivers, which otherwise swamp the tire-age signal).
    clean["driver_baseline"] = clean.groupby(
        ["circuit", "year", "Driver"]
    )["relative_time"].transform("mean")
    clean["driver_relative_time"] = clean["relative_time"] - clean["driver_baseline"]

    curves = {}
    for (circuit, compound), group in clean.groupby(["circuit", "Compound"]):
        if len(group) < min_laps:
            print(f"  [skip] {circuit}/{compound}: only {len(group)} laps")
            continue

        X = pd.DataFrame({
            "TyreLife": group["TyreLife"],
            "tyre_age_sq": group["TyreLife"] ** 2,
        })
        y = group["driver_relative_time"].values

        model = LinearRegression().fit(X, y)
        preds = model.predict(X)
        deg_rate, deg_accel = model.coef_

        curves[(circuit, compound)] = DegradationCurve(
            circuit=circuit, compound=compound,
            base_deviation=float(model.intercept_),
            deg_rate=float(deg_rate),
            deg_accel=float(deg_accel),
            n_laps_fit=len(group),
            mae_seconds=float(mean_absolute_error(y, preds)),
            r2=float(r2_score(y, preds)),
        )
        print(f"  [fit] {circuit}/{compound}: n={len(group)} "
              f"deg_rate={deg_rate:.3f}s/lap MAE={curves[(circuit,compound)].mae_seconds:.3f}s "
              f"R2={curves[(circuit,compound)].r2:.3f}")

    return curves


if __name__ == "__main__":
    laps = pd.read_parquet(PROCESSED_DIR / "laps_raw.parquet")
    curves = fit_degradation_curves(laps)
    rows = [vars(c) for c in curves.values()]
    pd.DataFrame(rows).to_parquet(PROCESSED_DIR / "degradation_curves.parquet", index=False)
    print(f"\nSaved {len(rows)} curves.")