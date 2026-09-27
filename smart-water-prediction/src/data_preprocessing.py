from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Date", "Time", "Consumption", "Temperature", "Rain", "Sun"]
FEATURE_COLUMNS = [
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
TARGET_COLUMN = "Consumption"


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    """Load dataset from CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file was not found at: {csv_path}")
    return pd.read_csv(csv_path)


def inspect_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Collect key dataset diagnostics for reporting."""
    unique_values = {}
    for column in ["Date", "Time"]:
        if column in df.columns:
            unique_values[column] = int(df[column].nunique(dropna=True))

    numeric_description = df.describe(include=[np.number]).transpose()
    numeric_description = numeric_description.reset_index().rename(columns={"index": "column"})

    return {
        "shape": df.shape,
        "head": df.head(),
        "tail": df.tail(),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_timestamps": int(df.duplicated(subset=[c for c in ["Date", "Time"] if c in df.columns]).sum())
        if {"Date", "Time"}.issubset(df.columns)
        else None,
        "numeric_description": numeric_description,
        "unique_values": unique_values,
    }


def _parse_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    combined = date_series.astype(str).str.strip() + " " + time_series.astype(str).str.strip()

    parsed_default = pd.to_datetime(combined, errors="coerce")
    parsed_dayfirst = pd.to_datetime(combined, errors="coerce", dayfirst=True)

    if parsed_dayfirst.notna().sum() > parsed_default.notna().sum():
        return parsed_dayfirst
    return parsed_default


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean, and engineer features from Date/Time."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    data = df.copy()
    data["DateTime"] = _parse_datetime(data["Date"], data["Time"])

    invalid_datetime_rows = data["DateTime"].isna().sum()
    if invalid_datetime_rows > 0:
        data = data.dropna(subset=["DateTime"]).copy()

    for column in ["Consumption", "Temperature", "Rain", "Sun"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["Consumption", "Temperature", "Rain", "Sun"]).copy()
    data = data.sort_values("DateTime").reset_index(drop=True)

    data["Year"] = data["DateTime"].dt.year
    data["Month"] = data["DateTime"].dt.month
    data["Day"] = data["DateTime"].dt.day
    data["Hour"] = data["DateTime"].dt.hour
    data["DayOfWeek"] = data["DateTime"].dt.dayofweek
    data["DayOfYear"] = data["DateTime"].dt.dayofyear
    data["WeekOfYear"] = data["DateTime"].dt.isocalendar().week.astype(int)
    data["IsWeekend"] = (data["DayOfWeek"] >= 5).astype(int)
    data["Quarter"] = data["DateTime"].dt.quarter

    data.attrs["invalid_datetime_rows_dropped"] = int(invalid_datetime_rows)
    return data


def build_feature_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model-ready features and target."""
    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()
    return X, y


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return a simple summary used in the app and training script."""
    if "DateTime" not in df.columns:
        raise ValueError("DateTime column not found. Run preprocess_data first.")

    return {
        "records": int(len(df)),
        "date_start": df["DateTime"].min().isoformat() if len(df) else None,
        "date_end": df["DateTime"].max().isoformat() if len(df) else None,
        "consumption_mean": float(df["Consumption"].mean()) if len(df) else None,
        "consumption_min": float(df["Consumption"].min()) if len(df) else None,
        "consumption_max": float(df["Consumption"].max()) if len(df) else None,
    }
