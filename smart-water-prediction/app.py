from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.data_preprocessing import (
    FEATURE_COLUMNS,
    build_feature_target,
    load_raw_data,
    preprocess_data,
    summarize_dataset,
)
from src.predict import load_feature_columns, load_model, validate_and_prepare_input


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "waterconsumption_noNaN_fixedTimechange.csv"
MODELS_DIR = ROOT_DIR / "models"


st.set_page_config(page_title="Smart Water Consumption Prediction", page_icon="💧", layout="wide")


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    raw = load_raw_data(DATA_PATH)
    return preprocess_data(raw)


@st.cache_data
def load_model_comparison() -> pd.DataFrame | None:
    path = MODELS_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_best_predictions() -> pd.DataFrame | None:
    path = MODELS_DIR / "best_model_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def render_prediction_tab(data: pd.DataFrame) -> None:
    st.subheader("Prediction")
    st.write("Use the sidebar inputs to predict hourly water consumption.")

    try:
        feature_order = load_feature_columns(MODELS_DIR)
        model = load_model(MODELS_DIR)
        model_ready = True
    except Exception as error:
        st.error(f"Model is not ready: {error}")
        model_ready = False
        feature_order = FEATURE_COLUMNS
        model = None

    if st.button("Predict Water Consumption", type="primary", use_container_width=True):
        sidebar_inputs = {
            "Temperature": st.session_state.temperature,
            "Rain": st.session_state.rain,
            "Sun": st.session_state.sun,
            "Hour": st.session_state.hour,
            "DayOfWeek": st.session_state.day_of_week,
            "Month": st.session_state.month,
            "DayOfYear": st.session_state.day_of_year,
            "IsWeekend": st.session_state.is_weekend,
            "Quarter": st.session_state.quarter,
        }

        if not model_ready:
            st.warning("Please train the model first: python src/train_model.py")
            return

        try:
            input_df = validate_and_prepare_input(sidebar_inputs, feature_order)
            prediction = float(model.predict(input_df)[0])
            st.metric("Predicted Water Consumption", f"{prediction:.3f}")
        except Exception as error:
            st.error(f"Prediction failed: {error}")

    with st.expander("Feature Order Used by the Model"):
        st.write(feature_order)


def render_data_overview_tab(data: pd.DataFrame) -> None:
    st.subheader("Data Overview")
    summary = summarize_dataset(data)

    c1, c2, c3 = st.columns(3)
    c1.metric("Number of Records", f"{summary['records']:,}")
    c2.metric("Average Consumption", f"{summary['consumption_mean']:.3f}")
    c3.metric("Minimum Consumption", f"{summary['consumption_min']:.3f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Maximum Consumption", f"{summary['consumption_max']:.3f}")
    c5.metric("Date Start", summary["date_start"])
    c6.metric("Date End", summary["date_end"])

    st.dataframe(data.head(20), use_container_width=True)


def render_analytics_tab(data: pd.DataFrame) -> None:
    st.subheader("EDA / Analytics")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data["Consumption"], kde=True, ax=ax)
        ax.set_title("Water Consumption Distribution")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=data, x="DateTime", y="Consumption", ax=ax)
        ax.set_title("Consumption Over Time")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        avg_hour = data.groupby("Hour", as_index=False)["Consumption"].mean()
        sns.barplot(data=avg_hour, x="Hour", y="Consumption", ax=ax, color="#2d7dd2")
        ax.set_title("Average Consumption by Hour")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        avg_day = data.groupby("DayOfWeek", as_index=False)["Consumption"].mean()
        sns.barplot(data=avg_day, x="DayOfWeek", y="Consumption", ax=ax, color="#00a6a6")
        ax.set_title("Average Consumption by Day of Week")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        avg_month = data.groupby("Month", as_index=False)["Consumption"].mean()
        sns.barplot(data=avg_month, x="Month", y="Consumption", ax=ax, color="#f18f01")
        ax.set_title("Average Consumption by Month")
        st.pyplot(fig)

        for feature in ["Temperature", "Rain", "Sun"]:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.scatterplot(data=data, x=feature, y="Consumption", ax=ax, alpha=0.6)
            ax.set_title(f"{feature} vs Consumption")
            st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        corr_cols = ["Consumption", "Temperature", "Rain", "Sun", "Hour", "DayOfWeek", "Month", "DayOfYear", "Quarter"]
        sns.heatmap(data[corr_cols].corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)



def render_model_performance_tab(data: pd.DataFrame) -> None:
    st.subheader("Model Performance")

    comparison = load_model_comparison()
    predictions = load_best_predictions()

    if comparison is None:
        st.warning("Model comparison not found. Run: python src/train_model.py")
        return

    st.dataframe(comparison, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=comparison, x="Model", y="RMSE", ax=ax, palette="viridis")
        ax.tick_params(axis="x", rotation=30)
        ax.set_title("RMSE Comparison")
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=comparison, x="Model", y="R2", ax=ax, palette="magma")
        ax.tick_params(axis="x", rotation=30)
        ax.set_title("R² Comparison")
        st.pyplot(fig)

    if predictions is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(predictions["Actual"], label="Actual")
        ax.plot(predictions["Predicted"], label="Predicted")
        ax.set_title("Actual vs Predicted (Best Model)")
        ax.legend()
        st.pyplot(fig)


def render_about_tab() -> None:
    st.subheader("About Project")
    st.markdown(
        """
        - This is a **regression** problem because consumption is a continuous numeric target.
        - Chronological splitting is used to avoid learning from future points.
        - Data leakage is prevented by excluding current target values from features.
        - Multiple models are compared because performance depends on pattern complexity.
        - **MAE** is the average absolute prediction error.
        - **RMSE** penalizes larger errors more strongly.
        - **R²** measures explained variance (closer to 1 is better).
        - Streamlit loads the trained model and feature metadata, then predicts from user inputs.
        """
    )


def main() -> None:
    st.title("💧 Smart Water Consumption Prediction & Analytics System")
    st.sidebar.header("Prediction Inputs")

    st.sidebar.number_input("Temperature", value=20.0, key="temperature")
    st.sidebar.number_input("Rain", value=0.0, key="rain")
    st.sidebar.number_input("Sun", value=0.0, key="sun")
    st.sidebar.slider("Hour", min_value=0, max_value=23, value=12, key="hour")
    st.sidebar.slider("Day of Week", min_value=0, max_value=6, value=2, key="day_of_week")
    st.sidebar.slider("Month", min_value=1, max_value=12, value=6, key="month")
    st.sidebar.slider("Day of Year", min_value=1, max_value=366, value=180, key="day_of_year")
    st.sidebar.selectbox("Weekend", options=[0, 1], key="is_weekend")
    st.sidebar.slider("Quarter", min_value=1, max_value=4, value=2, key="quarter")

    try:
        data = load_processed_data()
    except Exception as error:
        st.error(f"Failed to load dataset: {error}")
        st.info("Place waterconsumption_noNaN_fixedTimechange.csv in smart-water-prediction/data/")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Prediction", "Data Overview", "EDA / Analytics", "Model Performance", "About Project"]
    )

    with tab1:
        render_prediction_tab(data)
    with tab2:
        render_data_overview_tab(data)
    with tab3:
        render_analytics_tab(data)
    with tab4:
        render_model_performance_tab(data)
    with tab5:
        render_about_tab()


if __name__ == "__main__":
    main()
