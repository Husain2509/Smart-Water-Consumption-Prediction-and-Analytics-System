"""Notebook-equivalent analysis script for Smart Water Consumption project."""

from pathlib import Path

from src.data_preprocessing import inspect_dataset, load_raw_data, preprocess_data

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "waterconsumption_noNaN_fixedTimechange.csv"


def main() -> None:
    raw = load_raw_data(DATA_PATH)
    inspection = inspect_dataset(raw)
    print("Dataset shape:", inspection["shape"])
    print("Columns:", inspection["columns"])
    print("Missing values:\n", inspection["missing_values"])
    print("Duplicate rows:", inspection["duplicate_rows"])

    data = preprocess_data(raw)
    print("Processed shape:", data.shape)
    print(data.head())


if __name__ == "__main__":
    main()
