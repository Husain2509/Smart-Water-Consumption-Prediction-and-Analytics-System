from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.data_preprocessing import MODEL_FEATURES, load_dataset, preprocess_data
from src.predict import load_trained_model, predict_consumption

st.set_page_config(page_title="Smart Water Consumption Prediction", page_icon="💧", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "waterconsumption_noNaN_fixedTimechange.csv"


@st.cache_data
def get_processed_data(data_path: str):
    raw = load_dataset(data_path)
    return preprocess_data(raw)


@st.cache_data
def get_model_comparison():
    path = PROJECT_ROOT / "reports" / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def get_prediction_results():
    path = PROJECT_ROOT / "reports" / "best_model_predictions.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["Datetime"])
    return pd.DataFrame()


def main() -> None:
    st.title("💧 Smart Water Consumption Prediction & Analytics System")
    st.caption("End-to-end ML workflow for hourly water consumption forecasting using weather and time-based features.")

    st.sidebar.header("Prediction Inputs")
    temperature = st.sidebar.number_input("Temperature", value=25.0)
    rain = st.sidebar.number_input("Rain", value=0.0)
    sun = st.sidebar.number_input("Sun", value=6.0)
    hour = st.sidebar.slider("Hour", 0, 23, 12)
    day_of_week = st.sidebar.slider("Day of Week", 0, 6, 2)
    month = st.sidebar.slider("Month", 1, 12, 6)
    day_of_year = st.sidebar.slider("Day of Year", 1, 366, 180)
    is_weekend = st.sidebar.selectbox("Weekend", [0, 1], index=0)
    quarter = st.sidebar.selectbox("Quarter", [1, 2, 3, 4], index=1)

    tabs = st.tabs(["Prediction", "Data Overview", "EDA / Analytics", "Model Performance", "About Project"])

    with tabs[0]:
        st.subheader("Prediction")
        st.write("Use the sidebar to provide weather and time inputs, then click the prediction button.")
        if st.button("Predict Water Consumption", type="primary"):
            try:
                model_bundle = load_trained_model(PROJECT_ROOT / "models" / "best_model.pkl")
                payload = {
                    "Temperature": temperature,
                    "Rain": rain,
                    "Sun": sun,
                    "Hour": hour,
                    "DayOfWeek": day_of_week,
                    "Month": month,
                    "DayOfYear": day_of_year,
                    "IsWeekend": is_weekend,
                    "Quarter": quarter,
                }

                if model_bundle.get("feature_columns") != MODEL_FEATURES:
                    st.error("Model feature order does not match expected input order. Retrain the model.")
                else:
                    prediction = predict_consumption(payload, model_bundle=model_bundle)
                    st.metric("Predicted Water Consumption", f"{prediction:,.3f}")
                    st.info("Unit follows the same unit used in the dataset's Consumption column.")
            except FileNotFoundError:
                st.error("Model not found. Please train first using: python src/train_model.py")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    df = pd.DataFrame()
    try:
        if DEFAULT_DATA_PATH.exists():
            df = get_processed_data(str(DEFAULT_DATA_PATH))
    except Exception as exc:
        st.warning(f"Could not load dataset: {exc}")

    with tabs[1]:
        st.subheader("Data Overview")
        if df.empty:
            st.warning("Dataset not found in data/ or failed to parse.")
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Number of Records", f"{len(df):,}")
            c2.metric("Date Range (Start)", str(df["Datetime"].min().date()))
            c3.metric("Date Range (End)", str(df["Datetime"].max().date()))
            c4.metric("Average Consumption", f"{df['Consumption'].mean():.3f}")
            c5.metric("Min / Max", f"{df['Consumption'].min():.3f} / {df['Consumption'].max():.3f}")
            with st.expander("Preview Data"):
                st.dataframe(df.head(20), use_container_width=True)

    with tabs[2]:
        st.subheader("EDA / Analytics")
        if df.empty:
            st.warning("Load the dataset to view analytics.")
        else:
            sns.set_theme(style="whitegrid")
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df["Consumption"], kde=True, ax=ax, color="#1f77b4")
                ax.set_title("Water Consumption Distribution")
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                hourly = df.groupby("Hour", as_index=False)["Consumption"].mean()
                sns.lineplot(data=hourly, x="Hour", y="Consumption", marker="o", ax=ax)
                ax.set_title("Average Consumption by Hour")
                st.pyplot(fig)

            col3, col4 = st.columns(2)
            with col3:
                fig, ax = plt.subplots(figsize=(6, 4))
                dow = df.groupby("DayOfWeek", as_index=False)["Consumption"].mean()
                sns.barplot(
                    data=dow, x="DayOfWeek", y="Consumption", hue="DayOfWeek", legend=False, palette="viridis", ax=ax
                )
                ax.set_title("Average Consumption by Day of Week")
                st.pyplot(fig)
            with col4:
                fig, ax = plt.subplots(figsize=(6, 4))
                month = df.groupby("Month", as_index=False)["Consumption"].mean()
                sns.lineplot(data=month, x="Month", y="Consumption", marker="o", ax=ax)
                ax.set_title("Average Consumption by Month")
                st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(10, 4))
            monthly = (
                df.set_index("Datetime")["Consumption"].resample("M").mean().reset_index(name="Consumption")
            )
            sns.lineplot(data=monthly, x="Datetime", y="Consumption", marker="o", ax=ax)
            ax.set_title("Monthly Consumption Trend")
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(8, 5))
            corr_cols = ["Consumption", "Temperature", "Rain", "Sun", "Hour", "DayOfWeek", "Month", "DayOfYear"]
            sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

    with tabs[3]:
        st.subheader("Model Performance")
        comparison_df = get_model_comparison()
        pred_df = get_prediction_results()

        if comparison_df.empty:
            st.info("Model performance file not found. Run training first.")
        else:
            st.dataframe(comparison_df, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(
                    data=comparison_df, x="Model", y="RMSE", hue="Model", legend=False, palette="Blues_d", ax=ax
                )
                ax.tick_params(axis="x", rotation=20)
                ax.set_title("RMSE Comparison")
                st.pyplot(fig)
            with c2:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(
                    data=comparison_df, x="Model", y="R2", hue="Model", legend=False, palette="Greens_d", ax=ax
                )
                ax.tick_params(axis="x", rotation=20)
                ax.set_title("R² Comparison")
                st.pyplot(fig)

        if pred_df.empty:
            st.info("Prediction comparison file not found. Run training first.")
        else:
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(pred_df["Datetime"], pred_df["Actual"], label="Actual", linewidth=2)
            ax.plot(pred_df["Datetime"], pred_df["Predicted"], label="Predicted", linewidth=1.6)
            ax.set_title("Actual vs Predicted")
            ax.legend()
            st.pyplot(fig)

    with tabs[4]:
        st.subheader("About Project")
        st.markdown(
            """
- This is a **regression** problem because the target (Consumption) is a continuous numeric value.
- We use a **chronological split** so the model learns from past data and is tested on future-like data.
- Avoiding **data leakage** is important: the target value is never used as an input feature for the same row.
- Multiple models are compared to pick the best-performing algorithm on real evaluation metrics.
- **MAE**: average absolute error.
- **RMSE**: square-rooted error magnitude that penalizes larger mistakes more.
- **R²**: fraction of variance explained by the model.
- Streamlit provides an easy UI that loads the trained model and predicts from user inputs.
            """
        )


if __name__ == "__main__":
    main()
