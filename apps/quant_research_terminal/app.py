from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Path setup
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# -----------------------------
# Paths to project outputs
# -----------------------------
REGIME_DIR = PROJECT_ROOT / "research" / "market_regime_detection"
VOL_DIR = PROJECT_ROOT / "research" / "volatility_forecasting"
PAIRS_DIR = PROJECT_ROOT / "research" / "pairs_trading_cointegration"

REGIME_RESULTS = REGIME_DIR / "results"
VOL_RESULTS = VOL_DIR / "results"
PAIRS_RESULTS = PAIRS_DIR / "results"


# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Quant Research Terminal",
    page_icon="📈",
    layout="wide",
)


# -----------------------------
# Helper functions
# -----------------------------
def load_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def show_missing(path: Path):
    st.info(f"Output not found: `{path}`")


def metric_card(label: str, value: str):
    st.metric(label=label, value=value)


def project_status(path: Path) -> str:
    return "Ready" if path.exists() else "Missing Outputs"


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🧠 Quant Research Terminal")
st.sidebar.caption("Unified dashboard for quant research modules")

section = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Market Regime",
        "Volatility Forecasting",
        "Pairs Trading",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Status")

st.sidebar.write(f"🌍 Market Regime: **{project_status(REGIME_RESULTS / 'latest_regime.json')}**")
st.sidebar.write(f"📉 Volatility: **{project_status(VOL_RESULTS / 'summary.json')}**")
st.sidebar.write(f"🔗 Pairs Trading: **{project_status(PAIRS_RESULTS / 'summary.json')}**")


# -----------------------------
# Header
# -----------------------------
st.title("📈 Quant Research Terminal")
st.caption("Interactive dashboard for market regime detection, volatility forecasting, and statistical arbitrage research")


# -----------------------------
# Overview
# -----------------------------
if section == "Overview":
    st.subheader("Overview")

    regime_json = load_json(REGIME_RESULTS / "latest_regime.json")
    vol_summary = load_json(VOL_RESULTS / "summary.json")
    pairs_summary = load_json(PAIRS_RESULTS / "summary.json")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🌍 Market Regime")
        if regime_json:
            metric_card("Current Regime", regime_json.get("current_regime", "N/A"))
            st.write(f"**Date:** {regime_json.get('date', 'N/A')}")
            st.write(f"**Confidence:** {regime_json.get('confidence_label', 'N/A')}")
        else:
            st.info("Run the market regime project first.")

    with col2:
        st.markdown("### 📉 Volatility Forecasting")
        if vol_summary:
            metric_card("Best RMSE Model", vol_summary.get("best_model_by_rmse", "N/A"))
            metric_card("Best MAE Model", vol_summary.get("best_model_by_mae", "N/A"))
        else:
            st.info("Run the volatility forecasting project first.")

    with col3:
        st.markdown("### 🔗 Pairs Trading")
        if pairs_summary:
            metric_card("Pair", pairs_summary.get("pair_name", "N/A"))
            metric_card("Sharpe Ratio", f"{pairs_summary.get('sharpe_ratio', 0):.4f}")
        else:
            st.info("Run the pairs trading project first.")

    st.divider()

    st.subheader("Research Coverage")
    coverage_df = pd.DataFrame(
        {
            "Project": [
                "Market Regime Detection",
                "Volatility Forecasting",
                "Pairs Trading Cointegration",
            ],
            "Area": [
                "Regime Modeling",
                "Risk Modeling",
                "Statistical Arbitrage",
            ],
            "Status": [
                "Ready" if regime_json else "Missing Outputs",
                "Ready" if vol_summary else "Missing Outputs",
                "Ready" if pairs_summary else "Missing Outputs",
            ],
        }
    )
    st.dataframe(coverage_df, use_container_width=True, hide_index=True)


# -----------------------------
# Market Regime Section
# -----------------------------
elif section == "Market Regime":
    st.subheader("🌍 Market Regime Detection")

    regime_json = load_json(REGIME_RESULTS / "latest_regime.json")
    regime_chart = REGIME_RESULTS / "regime_chart.png"

    if regime_json:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card("Date", regime_json.get("date", "N/A"))
        with col2:
            metric_card("Current Regime", regime_json.get("current_regime", "N/A"))
        with col3:
            metric_card("Confidence", regime_json.get("confidence_label", "N/A"))
        with col4:
            metric_card("State ID", str(regime_json.get("state_id", "N/A")))

        st.markdown("### Interpretation")
        st.write(regime_json.get("interpretation", "No interpretation available."))
    else:
        show_missing(REGIME_RESULTS / "latest_regime.json")

    st.markdown("### Regime Chart")
    if regime_chart.exists():
        st.image(str(regime_chart), use_container_width=True)
    else:
        show_missing(regime_chart)


# -----------------------------
# Volatility Forecasting Section
# -----------------------------
elif section == "Volatility Forecasting":
    st.subheader("📉 Volatility Forecasting")

    metrics_path = VOL_RESULTS / "model_metrics.csv"
    summary_path = VOL_RESULTS / "summary.json"
    chart_path = VOL_RESULTS / "forecast_comparison.png"
    preds_path = VOL_RESULTS / "volatility_predictions.csv"

    metrics_df = load_csv(metrics_path)
    summary_json = load_json(summary_path)
    preds_df = load_csv(preds_path)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Best Models")
        if summary_json:
            metric_card("Best RMSE Model", summary_json.get("best_model_by_rmse", "N/A"))
            metric_card("Best MAE Model", summary_json.get("best_model_by_mae", "N/A"))
        else:
            show_missing(summary_path)

    with col2:
        st.markdown("### Model Metrics")
        if metrics_df is not None:
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        else:
            show_missing(metrics_path)

    st.markdown("### Forecast Comparison Chart")
    if chart_path.exists():
        st.image(str(chart_path), use_container_width=True)
    else:
        show_missing(chart_path)

    if preds_df is not None:
        st.markdown("### Interactive Forecast View")

        plot_df = preds_df.copy()
        if "Date" in plot_df.columns:
            plot_df["Date"] = pd.to_datetime(plot_df["Date"])

            value_cols = [c for c in plot_df.columns if c != "Date"]
            plot_long = plot_df.melt(
                id_vars="Date",
                value_vars=value_cols,
                var_name="Series",
                value_name="Value",
            )

            fig = px.line(
                plot_long,
                x="Date",
                y="Value",
                color="Series",
                title="Actual vs Predicted Volatility",
            )
            st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Pairs Trading Section
# -----------------------------
elif section == "Pairs Trading":
    st.subheader("🔗 Pairs Trading Cointegration")

    summary_path = PAIRS_RESULTS / "summary.json"
    zscore_chart = PAIRS_RESULTS / "spread_zscore.png"
    equity_chart = PAIRS_RESULTS / "equity_curve.png"
    backtest_path = PAIRS_RESULTS / "pairs_trading_backtest.csv"

    summary_json = load_json(summary_path)
    backtest_df = load_csv(backtest_path)

    if summary_json:
        col1, col2, col3 = st.columns(3)

        with col1:
            metric_card("Pair", summary_json.get("pair_name", "N/A"))
            metric_card("Cointegration p-value", f"{summary_json.get('cointegration_pvalue', 0):.6f}")

        with col2:
            metric_card("Sharpe Ratio", f"{summary_json.get('sharpe_ratio', 0):.4f}")
            metric_card("Win Rate", f"{summary_json.get('win_rate', 0):.2%}")

        with col3:
            metric_card("Total Return", f"{summary_json.get('total_return', 0):.2%}")
            metric_card("Max Drawdown", f"{summary_json.get('max_drawdown', 0):.2%}")
    else:
        show_missing(summary_path)

    st.markdown("### Z-Score and Equity Curve")
    c1, c2 = st.columns(2)

    with c1:
        if zscore_chart.exists():
            st.image(str(zscore_chart), use_container_width=True)
        else:
            show_missing(zscore_chart)

    with c2:
        if equity_chart.exists():
            st.image(str(equity_chart), use_container_width=True)
        else:
            show_missing(equity_chart)

    if backtest_df is not None:
        st.markdown("### Backtest Preview")
        st.dataframe(backtest_df.tail(20), use_container_width=True)