from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"


def load_feature_columns(model_dir: Path = MODELS_DIR) -> list[str]:
    feature_file = model_dir / "feature_columns.json"
    if not feature_file.exists():
        raise FileNotFoundError(
            "Feature metadata missing. Train the model first so feature_columns.json is created."
        )
    with open(feature_file, "r", encoding="utf-8") as file:
        return json.load(file)


def load_model(model_dir: Path = MODELS_DIR):
    model_path = model_dir / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Model file not found. Run training first to create best_model.pkl.")
    return joblib.load(model_path)


def validate_and_prepare_input(input_data: dict, feature_columns: list[str]) -> pd.DataFrame:
    missing_features = [feature for feature in feature_columns if feature not in input_data]
    if missing_features:
        raise ValueError(f"Missing input features: {missing_features}")

    row = {}
    for feature in feature_columns:
        value = input_data[feature]
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Feature '{feature}' must be numeric.") from error

        if pd.isna(value):
            raise ValueError(f"Feature '{feature}' cannot be NaN.")

        row[feature] = value

    return pd.DataFrame([row], columns=feature_columns)


def predict_consumption(input_data: dict) -> float:
    feature_columns = load_feature_columns()
    model = load_model()
    input_frame = validate_and_prepare_input(input_data, feature_columns)
    prediction = model.predict(input_frame)[0]
    return float(prediction)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict water consumption from feature inputs.")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--rain", type=float, required=True)
    parser.add_argument("--sun", type=float, required=True)
    parser.add_argument("--hour", type=float, required=True)
    parser.add_argument("--dayofweek", type=float, required=True)
    parser.add_argument("--month", type=float, required=True)
    parser.add_argument("--dayofyear", type=float, required=True)
    parser.add_argument("--isweekend", type=float, required=True)
    parser.add_argument("--quarter", type=float, required=True)
    return parser.parse_args()


def _cli() -> None:
    args = _parse_args()
    data = {
        "Temperature": args.temperature,
        "Rain": args.rain,
        "Sun": args.sun,
        "Hour": args.hour,
        "DayOfWeek": args.dayofweek,
        "Month": args.month,
        "DayOfYear": args.dayofyear,
        "IsWeekend": args.isweekend,
        "Quarter": args.quarter,
    }
    pred = predict_consumption(data)
    print(f"Predicted Water Consumption: {pred:.4f}")


if __name__ == "__main__":
    _cli()
