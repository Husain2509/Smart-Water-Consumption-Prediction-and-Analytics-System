from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from data_preprocessing import MODEL_FEATURES, build_feature_target, inspect_dataset, load_dataset, preprocess_data

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train water consumption prediction models.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/waterconsumption_noNaN_fixedTimechange.csv",
        help="Relative/absolute path to CSV dataset",
    )
    return parser.parse_args()


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def ensure_dirs(project_root: Path) -> None:
    for folder in ["models", "reports"]:
        (project_root / folder).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    ensure_dirs(project_root)

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = project_root / data_path

    raw_df = load_dataset(data_path)
    profile = inspect_dataset(raw_df)

    (project_root / "reports" / "data_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Loaded dataset from: {data_path}")
    print(f"Dataset shape: {raw_df.shape}")

    df = preprocess_data(raw_df)

    duplicate_timestamps = int(df.duplicated(subset=["Datetime"]).sum())
    print(f"Duplicate timestamps after parsing: {duplicate_timestamps}")

    X, y = build_feature_target(df)

    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    dt_train, dt_test = df["Datetime"].iloc[:split_index], df["Datetime"].iloc[split_index:]

    print("Using chronological 80/20 split (older data for training, newer data for testing).")

    models = {
        "Linear Regression": Pipeline(
            steps=[("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=12, min_samples_leaf=5),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=2
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=250, learning_rate=0.05),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
    else:
        print("XGBoost is not available in this environment; skipping XGBoost model.")

    records = []
    predictions_by_model: dict[str, np.ndarray] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions_by_model[model_name] = preds
        metrics = evaluate_model(y_test, preds)
        records.append({"Model": model_name, **metrics})

    results_df = pd.DataFrame(records).sort_values(by=["RMSE", "MAE"]).reset_index(drop=True)
    results_df.to_csv(project_root / "reports" / "model_comparison.csv", index=False)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]

    best_model_path = project_root / "models" / "best_model.pkl"
    joblib.dump(
        {
            "model_name": best_model_name,
            "feature_columns": MODEL_FEATURES,
            "model": best_model,
        },
        best_model_path,
    )

    scaler_path = project_root / "models" / "scaler.pkl"
    if isinstance(best_model, Pipeline) and "scaler" in best_model.named_steps:
        joblib.dump(best_model.named_steps["scaler"], scaler_path)
    elif scaler_path.exists():
        scaler_path.unlink()

    best_preds = predictions_by_model[best_model_name]
    pred_df = pd.DataFrame(
        {
            "Datetime": dt_test.values,
            "Actual": y_test.values,
            "Predicted": best_preds,
        }
    )
    pred_df.to_csv(project_root / "reports" / "best_model_predictions.csv", index=False)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="Model", y="RMSE", hue="Model", legend=False, palette="Blues_d")
    plt.xticks(rotation=20, ha="right")
    plt.title("Model RMSE Comparison")
    plt.tight_layout()
    plt.savefig(project_root / "reports" / "rmse_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(dt_test, y_test.values, label="Actual", linewidth=2)
    plt.plot(dt_test, best_preds, label="Predicted", linewidth=1.8)
    plt.title(f"Actual vs Predicted Consumption ({best_model_name})")
    plt.xlabel("Datetime")
    plt.ylabel("Consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(project_root / "reports" / "actual_vs_predicted.png", dpi=150)
    plt.close()

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(f"Saved model to: {best_model_path}")


if __name__ == "__main__":
    main()
