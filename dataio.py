"""Dataset loading and tidying for the choke-control step test.

The provided dataset is a 120-hour open-loop step test of a single naturally
flowing oil well. Control interval Ts = 1 hr. The raw columns are:

    Time_hr, Choke_pct, OilRate_bbl_hr, WHP_psi, FLP_psi, BHP_psi

Throughout the rest of the project we use short canonical names:

    t   -> Time_hr        (hr)
    u   -> Choke_pct      (%)
    Q   -> OilRate_bbl_hr (bbl/hr)
    WHP -> WHP_psi        (psi)   wellhead pressure
    FLP -> FLP_psi        (psi)   flowline pressure
    BHP -> BHP_psi        (psi)   bottomhole pressure
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CSV = DATA_DIR / "Autonomous_Choke_Control_Simulated_Dataset.csv"

# Raw dataset column names, in the order we rely on.
RAW_COLUMNS = [
    "Time_hr",
    "Choke_pct",
    "OilRate_bbl_hr",
    "WHP_psi",
    "FLP_psi",
    "BHP_psi",
]

# Map raw column -> canonical short name used across the codebase.
CANONICAL = {
    "Time_hr": "t",
    "Choke_pct": "u",
    "OilRate_bbl_hr": "Q",
    "WHP_psi": "WHP",
    "FLP_psi": "FLP",
    "BHP_psi": "BHP",
}

# The four measured process outputs the controller reasons about.
OUTPUTS = ["Q", "WHP", "FLP", "BHP"]


def load_step_test(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """Load the step-test CSV, validate it, and return a tidy frame.

    The returned frame keeps the raw column names, is sorted by time, and is
    guaranteed to contain every required column with no missing values. We
    fail loudly on any structural problem rather than silently returning a
    partial frame that would corrupt the identification downstream.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Step-test dataset not found at {path}. It should be committed at "
            f"{DEFAULT_CSV.relative_to(DATA_DIR.parent)}."
        )

    df = pd.read_csv(path)

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset {path.name} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}."
        )

    df = df[RAW_COLUMNS].copy()

    if df.isnull().any().any():
        bad = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Dataset {path.name} has missing values in columns: {bad}.")

    df = df.sort_values("Time_hr").reset_index(drop=True)
    return df


def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical short column names (t, u, Q, WHP, FLP, BHP)."""
    return df.rename(columns=CANONICAL)


def infer_sample_time(df: pd.DataFrame) -> float:
    """Infer the sampling interval (hr) from the Time_hr column.

    Raises if the sampling is not uniform, because the ARX(1) identification and
    the time-constant conversion both assume a fixed Ts.
    """
    dt = df["Time_hr"].diff().dropna().to_numpy()
    if len(dt) == 0:
        raise ValueError("Cannot infer sample time from a single-row dataset.")
    ts = float(dt[0])
    if not (dt == ts).all():
        raise ValueError(
            f"Non-uniform sampling detected in Time_hr (steps: {sorted(set(dt))}). "
            "ARX(1) identification assumes a fixed sample interval."
        )
    return ts


if __name__ == "__main__":
    frame = load_step_test()
    print(f"Loaded {len(frame)} rows from {DEFAULT_CSV.name}")
    print(f"Inferred Ts = {infer_sample_time(frame)} hr")
    print("\nChoke schedule (step changes):")
    steps = frame[frame["Choke_pct"].diff().fillna(1) != 0]
    for _, row in steps.iterrows():
        print(f"  t={int(row['Time_hr']):>3} hr  ->  choke {row['Choke_pct']:.0f}%")
    print("\nHead:")
    print(frame.head().to_string(index=False))
