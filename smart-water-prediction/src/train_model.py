from __future__ import annotations

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
from xgboost import XGBRegressor

from data_preprocessing import (
    FEATURE_COLUMNS,
    build_feature_target,
    inspect_dataset,
    load_raw_data,
    preprocess_data,
    summarize_dataset,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "waterconsumption_noNaN_fixedTimechange.csv"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"


def _ensure_directories() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _print_inspection(inspection: dict) -> None:
    print("\n===== DATASET INSPECTION =====")
    print(f"Shape: {inspection['shape']}")
    print("\nFirst 5 rows:")
    print(inspection["head"])
    print("\nLast 5 rows:")
    print(inspection["tail"])
    print("\nColumns:")
    print(inspection["columns"])
    print("\nData types:")
    print(pd.Series(inspection["dtypes"]))
    print("\nMissing values:")
    print(pd.Series(inspection["missing_values"]))
    print(f"\nDuplicate rows: {inspection['duplicate_rows']}")
    print(f"Duplicate timestamps (Date+Time): {inspection['duplicate_timestamps']}")
    print("\nUnique values (selected columns):")
    print(pd.Series(inspection["unique_values"]))
    print("\nDescriptive statistics:")
    print(inspection["numeric_description"])


def create_eda_plots(data: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.histplot(data["Consumption"], kde=True, bins=30)
    plt.title("Water Consumption Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_consumption_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=data, x="DateTime", y="Consumption")
    plt.title("Consumption Over Time")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_consumption_over_time.png", dpi=200)
    plt.close()

    hourly = data.groupby("Hour", as_index=False)["Consumption"].mean()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=hourly, x="Hour", y="Consumption", color="#2d7dd2")
    plt.title("Average Consumption by Hour")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_avg_consumption_by_hour.png", dpi=200)
    plt.close()

    dow = data.groupby("DayOfWeek", as_index=False)["Consumption"].mean()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=dow, x="DayOfWeek", y="Consumption", color="#00a6a6")
    plt.title("Average Consumption by Day of Week")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_avg_consumption_by_dayofweek.png", dpi=200)
    plt.close()

    monthly = data.groupby("Month", as_index=False)["Consumption"].mean()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=monthly, x="Month", y="Consumption", color="#f18f01")
    plt.title("Average Consumption by Month")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_avg_consumption_by_month.png", dpi=200)
    plt.close()

    for index, feature in enumerate(["Temperature", "Rain", "Sun"], start=6):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x=feature, y="Consumption", alpha=0.6)
        plt.title(f"{feature} vs Consumption")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{index:02d}_{feature.lower()}_vs_consumption.png", dpi=200)
        plt.close()

    corr_features = ["Consumption", "Temperature", "Rain", "Sun", "Hour", "DayOfWeek", "Month", "DayOfYear", "Quarter"]
    plt.figure(figsize=(10, 8))
    corr = data[corr_features].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_correlation_heatmap.png", dpi=200)
    plt.close()

    monthly_trend = (
        data.set_index("DateTime")["Consumption"].resample("M").mean().reset_index(name="Consumption")
    )
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=monthly_trend, x="DateTime", y="Consumption", marker="o")
    plt.title("Monthly Consumption Trend")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_monthly_consumption_trend.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=hourly, x="Hour", y="Consumption", marker="o")
    plt.title("Hourly Consumption Pattern")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "11_hourly_consumption_pattern.png", dpi=200)
    plt.close()


def evaluate_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    models = {
        "Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=12, min_samples_leaf=2, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            max_depth=None,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
    }

    rows = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        rows.append({"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})
        predictions[name] = {"model": model, "y_pred": y_pred}

    comparison = pd.DataFrame(rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    best_name = comparison.iloc[0]["Model"]
    best_model = predictions[best_name]["model"]
    best_pred = predictions[best_name]["y_pred"]

    return comparison, best_name, best_model, best_pred


def save_model_outputs(
    processed_data: pd.DataFrame,
    y_test: pd.Series,
    best_pred: np.ndarray,
    comparison: pd.DataFrame,
    best_name: str,
    best_model,
) -> None:
    comparison.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    scaler_path = MODELS_DIR / "scaler.pkl"
    if hasattr(best_model, "named_steps") and "scaler" in best_model.named_steps:
        joblib.dump(best_model.named_steps["scaler"], scaler_path)
    elif scaler_path.exists():
        scaler_path.unlink()

    with open(MODELS_DIR / "feature_columns.json", "w", encoding="utf-8") as file:
        json.dump(FEATURE_COLUMNS, file, indent=2)

    with open(MODELS_DIR / "best_model_info.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_model": best_name,
                "chronological_split": "80% train / 20% test",
                "note": "Split is chronological to avoid leakage from future data.",
            },
            file,
            indent=2,
        )

    dataset_summary = summarize_dataset(processed_data)
    with open(MODELS_DIR / "data_summary.json", "w", encoding="utf-8") as file:
        json.dump(dataset_summary, file, indent=2)

    pd.DataFrame({"Actual": y_test.values, "Predicted": best_pred}).to_csv(
        MODELS_DIR / "best_model_predictions.csv", index=False
    )


def create_model_plots(comparison: pd.DataFrame, y_test: pd.Series, best_pred: np.ndarray) -> None:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=comparison, x="Model", y="RMSE", palette="viridis")
    plt.xticks(rotation=30, ha="right")
    plt.title("Model RMSE Comparison")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "12_model_rmse_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=comparison, x="Model", y="R2", palette="magma")
    plt.xticks(rotation=30, ha="right")
    plt.title("Model R² Comparison")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "13_model_r2_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label="Actual", linewidth=2)
    plt.plot(best_pred, label="Predicted", linewidth=2)
    plt.title("Actual vs Predicted (Best Model)")
    plt.xlabel("Test Index")
    plt.ylabel("Consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "14_actual_vs_predicted.png", dpi=200)
    plt.close()


def run_training() -> None:
    _ensure_directories()

    raw_data = load_raw_data(DATA_PATH)
    inspection = inspect_dataset(raw_data)
    _print_inspection(inspection)

    processed_data = preprocess_data(raw_data)
    if len(processed_data) < 20:
        raise ValueError("Not enough valid rows after preprocessing to train a reliable model.")

    create_eda_plots(processed_data)

    X, y = build_feature_target(processed_data)
    split_index = int(len(processed_data) * 0.8)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    comparison, best_name, best_model, best_pred = evaluate_models(X_train, y_train, X_test, y_test)

    save_model_outputs(processed_data, y_test, best_pred, comparison, best_name, best_model)
    create_model_plots(comparison, y_test, best_pred)

    print("\n===== MODEL COMPARISON =====")
    print(comparison)
    print(f"\nBest model selected: {best_name}")
    print(f"Model artifacts saved to: {MODELS_DIR}")
    print(f"Plots saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    run_training()
