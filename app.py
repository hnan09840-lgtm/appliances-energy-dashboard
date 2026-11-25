import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ---------------------------------------------------------
# Smart Lecture? No: IoT Sensor Data Prediction Dashboard
# Project: Appliances Energy Prediction
# Inputs: T1 (kitchen temperature), RH_1 (kitchen humidity)
# Model: Linear Regression
# ---------------------------------------------------------

st.set_page_config(
    page_title="Appliances Energy Prediction Dashboard",
    page_icon="🌡️",
    layout="wide"
)

DATA_FILE = Path("cleaned_dataset.csv")
PREDICTIONS_FILE = Path("predictions.csv")

st.title("🌡️ Appliances Energy Prediction Dashboard")
st.caption(
    "IoT sensor data monitoring and next-step kitchen temperature prediction "
    "using Linear Regression."
)

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "T1", "RH_1"]).sort_values("date")
    return data

@st.cache_data
def load_predictions():
    results = pd.read_csv(PREDICTIONS_FILE)
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results = results.dropna(
        subset=["date", "Actual Temperature", "Predicted Temperature"]
    ).sort_values("date")
    return results

missing = [str(p) for p in [DATA_FILE, PREDICTIONS_FILE] if not p.exists()]
if missing:
    st.error(
        "The dashboard cannot find these required files: "
        + ", ".join(missing)
        + ". Place app.py, cleaned_dataset.csv, and predictions.csv in the same folder."
    )
    st.stop()

df = load_data()
pred = load_predictions()

if df.empty or pred.empty:
    st.error("One of the CSV files contains no usable rows.")
    st.stop()

section = st.sidebar.radio(
    "Dashboard Sections",
    ["Overview", "Temperature", "Humidity", "Prediction", "Evaluation"]
)
st.sidebar.markdown("---")
st.sidebar.write("**Dataset records:**", f"{len(df):,}")
st.sidebar.write("**Prediction records:**", f"{len(pred):,}")
st.sidebar.caption("Source: project cleaned dataset and prediction results.")

if section == "Overview":
    st.header("Overview")
    st.write("A summary of the kitchen sensor readings in the cleaned dataset.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Temperature", f"{df['T1'].mean():.2f} °C")
    c2.metric("Average Humidity", f"{df['RH_1'].mean():.2f} %")
    c3.metric("Total Sensor Readings", f"{len(df):,}")

    c4, c5 = st.columns(2)
    c4.metric("Minimum Temperature", f"{df['T1'].min():.2f} °C")
    c5.metric("Maximum Temperature", f"{df['T1'].max():.2f} °C")

    st.subheader("Sensor Data Preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.subheader("Temperature and Humidity Summary")
    summary = pd.DataFrame({
        "Measurement": ["Temperature (°C)", "Humidity (%)"],
        "Average": [df["T1"].mean(), df["RH_1"].mean()],
        "Minimum": [df["T1"].min(), df["RH_1"].min()],
        "Maximum": [df["T1"].max(), df["RH_1"].max()]
    })
    st.dataframe(summary.round(2), use_container_width=True, hide_index=True)

elif section == "Temperature":
    st.header("Temperature Monitoring")
    st.write("Kitchen temperature (T1) over time.")
    fig = px.line(
        df, x="date", y="T1",
        labels={"date": "Date and Time", "T1": "Temperature (°C)"},
        title="Kitchen Temperature Over Time"
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Average Kitchen Temperature", f"{df['T1'].mean():.2f} °C")

elif section == "Humidity":
    st.header("Humidity Monitoring")
    st.write("Kitchen relative humidity (RH_1) over time.")
    fig = px.line(
        df, x="date", y="RH_1",
        labels={"date": "Date and Time", "RH_1": "Relative Humidity (%)"},
        title="Kitchen Relative Humidity Over Time"
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Average Kitchen Humidity", f"{df['RH_1'].mean():.2f} %")

elif section == "Prediction":
    st.header("Temperature Prediction")
    st.write(
        "Comparison between the actual next temperature reading and the "
        "temperature predicted by the Linear Regression model."
    )
    fig = px.line(
        pred,
        x="date",
        y=["Actual Temperature", "Predicted Temperature"],
        labels={
            "date": "Date and Time",
            "value": "Temperature (°C)",
            "variable": "Reading"
        },
        title="Actual vs Predicted Temperature"
    )
    fig.update_layout(hovermode="x unified", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(pred.head(10), use_container_width=True, hide_index=True)

elif section == "Evaluation":
    st.header("Model Performance Evaluation")
    st.write("The following metrics are calculated directly from predictions.csv.")

    mae = (pred["Actual Temperature"] - pred["Predicted Temperature"]).abs().mean()
    mse = (
        (pred["Actual Temperature"] - pred["Predicted Temperature"]) ** 2
    ).mean()
    actual_mean = pred["Actual Temperature"].mean()
    ss_res = (
        (pred["Actual Temperature"] - pred["Predicted Temperature"]) ** 2
    ).sum()
    ss_tot = ((pred["Actual Temperature"] - actual_mean) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.4f}")
    c2.metric("MSE", f"{mse:.4f}")
    c3.metric("R² Score", f"{r2:.4f}")

    st.markdown("**Metric meaning**")
    st.markdown(
        "- **MAE:** Average absolute difference between actual and predicted temperatures.\n"
        "- **MSE:** Average squared prediction error.\n"
        "- **R²:** Indicates how well the model explains variation in the target values."
    )

    st.caption(
        "Model: Linear Regression | Input features: T1 and RH_1 | "
        "Target: Next_T1 | Train/test split: chronological 80/20."
    )

st.markdown("---")
st.caption("Appliances Energy Prediction | IT351 Project")
