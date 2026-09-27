from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = ["Date", "Time", "Consumption", "Temperature", "Rain", "Sun"]
MODEL_FEATURES = [
    "Temperature",
    "Rain",
    "Sun",
    "Hour",
    "DayOfWeek",
    "Month",
    "DayOfYear",
    "IsWeekend",
    "Quarter",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    return _normalize_columns(df)


def inspect_dataset(df: pd.DataFrame) -> Dict[str, object]:
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]

    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_expected_columns": missing_columns,
        "head": df.head(5).to_dict(orient="records"),
        "tail": df.tail(5).to_dict(orient="records"),
    }

    numeric_cols = [c for c in ["Consumption", "Temperature", "Rain", "Sun"] if c in df.columns]
    if numeric_cols:
        info["describe"] = df[numeric_cols].describe().to_dict()

    if "Date" in df.columns:
        info["unique_date_count"] = int(df["Date"].nunique(dropna=True))
    if "Time" in df.columns:
        info["unique_time_count"] = int(df["Time"].nunique(dropna=True))

    return info


def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)

    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

    data = df.copy()
    data = _safe_numeric(data, ["Consumption", "Temperature", "Rain", "Sun"])

    date_str = data["Date"].astype(str).str.strip()
    time_str = data["Time"].astype(str).str.strip()
    data["Datetime"] = pd.to_datetime(date_str + " " + time_str, errors="coerce")

    if data["Datetime"].isna().all():
        raise ValueError("Could not parse Date and Time into datetime values.")

    data = data.dropna(subset=["Datetime", "Consumption", "Temperature", "Rain", "Sun"]).copy()
    data = data.sort_values("Datetime").reset_index(drop=True)

    data["Year"] = data["Datetime"].dt.year
    data["Month"] = data["Datetime"].dt.month
    data["Day"] = data["Datetime"].dt.day
    data["Hour"] = data["Datetime"].dt.hour
    data["DayOfWeek"] = data["Datetime"].dt.dayofweek
    data["DayOfYear"] = data["Datetime"].dt.dayofyear
    data["WeekOfYear"] = data["Datetime"].dt.isocalendar().week.astype(int)
    data["IsWeekend"] = (data["DayOfWeek"] >= 5).astype(int)
    data["Quarter"] = data["Datetime"].dt.quarter

    return data


def build_feature_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in MODEL_FEATURES + ["Consumption"] if c not in df.columns]
    if missing:
        raise ValueError(f"Required feature/target columns are missing after preprocessing: {missing}")

    X = df[MODEL_FEATURES].copy()
    y = df["Consumption"].copy()
    return X, y
