from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd


def load_trained_model(model_path: str | Path = "models/best_model.pkl") -> dict:
    path = Path(model_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def predict_consumption(input_data: Dict[str, float], model_bundle: dict | None = None) -> float:
    bundle = model_bundle or load_trained_model()

    feature_columns = bundle["feature_columns"]
    model = bundle["model"]

    missing = [f for f in feature_columns if f not in input_data]
    if missing:
        raise ValueError(f"Missing required input features: {missing}")

    row = pd.DataFrame([[input_data[col] for col in feature_columns]], columns=feature_columns)
    row = row.apply(pd.to_numeric, errors="coerce")

    if row.isna().any().any():
        raise ValueError("Input contains invalid values that cannot be converted to numbers.")

    pred = model.predict(row)
    return float(np.asarray(pred)[0])
