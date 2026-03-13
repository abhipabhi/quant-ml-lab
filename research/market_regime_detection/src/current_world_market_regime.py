from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import json
from pathlib import Path
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


@dataclass
class RegimeResult:
    current_regime: str
    confidence: float
    latest_date: str
    state_id: int
    summary_table: pd.DataFrame
    transition_matrix: pd.DataFrame


class WorldMarketRegimeDetector:
    """
    Fetches global market data, fits an HMM, and classifies the latest
    market regime into Bull / Bear / Risk-Off / Sideways.
    """

    def __init__(
        self,
        tickers: Dict[str, str] | None = None,
        period: str = "10y",
        interval: str = "1d",
        n_states: int = 4,
        random_state: int = 42,
    ) -> None:
        self.tickers = tickers or {
            "US": "^GSPC",         # S&P 500
            "EU": "^STOXX50E",     # Euro Stoxx 50
            "INDIA": "^NSEI",      # Nifty 50
            "JAPAN": "^N225",      # Nikkei 225
            "VIX": "^VIX",         # Volatility proxy
        }
        self.period = period
        self.interval = interval
        self.n_states = n_states
        self.random_state = random_state

        self.raw_data: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None
        self.model: GaussianHMM | None = None
        self.state_stats: pd.DataFrame | None = None
        self.state_labels: Dict[int, str] | None = None

    def fetch_data(self) -> pd.DataFrame:
        close_frames: List[pd.Series] = []

        for market_name, ticker in self.tickers.items():
            df = yf.download(
                ticker,
                period=self.period,
                interval=self.interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if df.empty:
                raise ValueError(f"Failed to fetch usable data for {market_name} ({ticker})")

            # Handle both possible yfinance formats:
            # 1) normal columns: Open, High, Low, Close, Volume
            # 2) MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                close_data = df.xs("Close", axis=1, level=0)
                if close_data.empty:
                    raise ValueError(f"No Close data found for {market_name} ({ticker})")
                series = close_data.iloc[:, 0].rename(market_name)
            else:
                if "Close" not in df.columns:
                    raise ValueError(f"No Close data found for {market_name} ({ticker})")
                series = df["Close"].rename(market_name)

            close_frames.append(series)

        prices = pd.concat(close_frames, axis=1).dropna()

        if prices.empty:
            raise ValueError("No overlapping price history found across selected markets.")

        self.raw_data = prices
        return prices

    def build_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        log_returns = np.log(prices / prices.shift(1)).dropna()

        feat = pd.DataFrame(index=log_returns.index)

        # Regional average equity return
        equity_cols = [c for c in log_returns.columns if c != "VIX"]
        feat["global_equity_return"] = log_returns[equity_cols].mean(axis=1)

        # Cross-market dispersion: risk fragmentation proxy
        feat["cross_sectional_dispersion"] = log_returns[equity_cols].std(axis=1)

        # Rolling realized volatility
        feat["realized_vol_20"] = feat["global_equity_return"].rolling(20).std()

        # Trend proxy
        avg_price = prices[equity_cols].mean(axis=1)
        ma_20 = avg_price.rolling(20).mean()
        ma_100 = avg_price.rolling(100).mean()
        feat["trend_ratio"] = ma_20 / ma_100 - 1.0

        # VIX level and change
        if "VIX" in prices.columns:
            feat["vix_level"] = prices["VIX"]
            feat["vix_return_5"] = np.log(prices["VIX"] / prices["VIX"].shift(5))
        else:
            feat["vix_level"] = 0.0
            feat["vix_return_5"] = 0.0

        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

        # Standardize for HMM stability
        standardized = (feat - feat.mean()) / feat.std(ddof=0)
        standardized = standardized.dropna()

        self.features = standardized
        return standardized

    def fit_hmm(self, features: pd.DataFrame) -> Tuple[GaussianHMM, np.ndarray]:
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=500,
            random_state=self.random_state,
        )
        model.fit(features.values)
        states = model.predict(features.values)

        self.model = model
        return model, states

    def characterize_states(
        self,
        features_unscaled: pd.DataFrame,
        states: np.ndarray,
    ) -> Tuple[pd.DataFrame, Dict[int, str]]:
        temp = features_unscaled.loc[features_unscaled.index[-len(states):]].copy()
        temp["state"] = states

        stats = (
            temp.groupby("state")
            .agg(
                avg_return=("global_equity_return", "mean"),
                avg_vol=("realized_vol_20", "mean"),
                avg_trend=("trend_ratio", "mean"),
                avg_vix=("vix_level", "mean"),
                count=("state", "size"),
            )
            .sort_index()
        )

        labels: Dict[int, str] = {}
        median_vol = stats["avg_vol"].median()
        median_vix = stats["avg_vix"].median()

        for state_id, row in stats.iterrows():
            ret = row["avg_return"]
            vol = row["avg_vol"]
            trend = row["avg_trend"]
            vix = row["avg_vix"]

            if ret < 0 and trend < 0 and (vol > median_vol or vix > median_vix):
                labels[state_id] = "Bear / Risk-Off"

            elif ret > 0 and trend > 0.02 and vol <= median_vol and vix <= median_vix:
                labels[state_id] = "Bull / Low-Vol"

            elif ret > 0 and trend > 0.02 and (vol > median_vol or vix > median_vix):
                labels[state_id] = "Bull / High-Vol"

            elif ret >= 0 and trend >= 0:
                labels[state_id] = "Neutral / Stable"

            else:
                labels[state_id] = "Transition / Mixed"

        self.state_stats = stats
        self.state_labels = labels
        return stats, labels

    def build_interpretation(self, regime: str) -> str:
        interpretations = {
            "Bull / Low-Vol": "Markets are trending upward with relatively low volatility and stable risk appetite.",
            "Bull / High-Vol": "Markets are positive but volatile, indicating risk-on conditions with elevated uncertainty.",
            "Neutral / Stable": "Markets appear range-bound or mildly positive, with relatively stable conditions.",
            "Bear / Risk-Off": "Markets are under stress, with negative returns and elevated volatility indicating risk-off behavior.",
            "Transition / Mixed": "Markets are showing mixed signals, suggesting a transition between broader regimes.",
        }
        return interpretations.get(regime, "No interpretation available for the current regime.")


    def confidence_label(self, confidence: float) -> str:
        if confidence >= 0.995:
            return "Very High"
        elif confidence >= 0.80:
            return "High"
        elif confidence >= 0.60:
            return "Moderate"
        return "Low"


    def save_json_output(self, result: RegimeResult, output_dir: str = "results") -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        payload = {
            "date": result.latest_date,
            "current_regime": result.current_regime,
            "state_id": result.state_id,
            "confidence": round(result.confidence, 6),
            "confidence_label": self.confidence_label(result.confidence),
            "interpretation": self.build_interpretation(result.current_regime),
        }

        with open(Path(output_dir) / "latest_regime.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)


    def plot_regimes(
        self,
        prices: pd.DataFrame,
        features_scaled: pd.DataFrame,
        states: np.ndarray,
        labels: Dict[int, str],
        output_dir: str = "results",
    ) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        equity_cols = [c for c in prices.columns if c != "VIX"]
        global_index = prices[equity_cols].mean(axis=1)
        global_index = global_index.loc[features_scaled.index]

        state_series = pd.Series(states, index=features_scaled.index, name="state")

        color_map = {
            "Bull / Low-Vol": "#2ca02c",
            "Bull / High-Vol": "#ff7f0e",
            "Neutral / Stable": "#1f77b4",
            "Bear / Risk-Off": "#d62728",
            "Transition / Mixed": "#9467bd",
        }

        plt.figure(figsize=(14, 7))
        plt.plot(global_index.index, global_index.values, color="black", linewidth=1.5, label="Global Equity Index")

        for state_id in sorted(state_series.unique()):
            mask = state_series == state_id
            regime_name = labels.get(int(state_id), f"State {state_id}")
            plt.scatter(
                global_index.index[mask],
                global_index[mask],
                s=10,
                label=regime_name,
                color=color_map.get(regime_name, "gray"),
                alpha=0.7,
            )

        plt.title("World Market Regime Detection")
        plt.xlabel("Date")
        plt.ylabel("Average Global Equity Index Level")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "regime_chart.png", dpi=300)
        plt.close()

    def detect(self) -> RegimeResult:
        prices = self.fetch_data()
        features_scaled = self.build_features(prices)

        log_returns = np.log(prices / prices.shift(1)).dropna()
        equity_cols = [c for c in log_returns.columns if c != "VIX"]

        interp = pd.DataFrame(index=log_returns.index)
        interp["global_equity_return"] = log_returns[equity_cols].mean(axis=1)
        interp["cross_sectional_dispersion"] = log_returns[equity_cols].std(axis=1)
        interp["realized_vol_20"] = interp["global_equity_return"].rolling(20).std()

        avg_price = prices[equity_cols].mean(axis=1)
        interp["trend_ratio"] = (
            avg_price.rolling(20).mean() / avg_price.rolling(100).mean() - 1.0
        )
        interp["vix_level"] = prices["VIX"] if "VIX" in prices.columns else 0.0
        interp["vix_return_5"] = (
            np.log(prices["VIX"] / prices["VIX"].shift(5)) if "VIX" in prices.columns else 0.0
        )
        interp = interp.replace([np.inf, -np.inf], np.nan).dropna()

        aligned_interp = interp.loc[features_scaled.index]

        model, states = self.fit_hmm(features_scaled)
        stats, labels = self.characterize_states(aligned_interp, states)

        latest_state = int(states[-1])
        latest_probs = model.predict_proba(features_scaled.values)[-1]
        confidence = float(latest_probs[latest_state])

        state_names = [labels.get(i, f"State {i}") for i in range(self.n_states)]

        transition_matrix = pd.DataFrame(
            model.transmat_,
            index=state_names,
            columns=state_names,
        )

        summary = stats.copy()
        summary["regime"] = summary.index.map(labels)
        summary = summary.reset_index().rename(columns={"state": "state_id"})
        summary = summary[["state_id", "regime", "avg_return", "avg_vol", "avg_trend", "avg_vix", "count"]]

        result = RegimeResult(
            current_regime=labels[latest_state],
            confidence=confidence,
            latest_date=str(features_scaled.index[-1].date()),
            state_id=latest_state,
            summary_table=summary,
            transition_matrix=transition_matrix,
        )

        self.save_json_output(result)
        self.plot_regimes(prices, features_scaled, states, labels)

        return result

def main() -> None:
    detector = WorldMarketRegimeDetector()
    result = detector.detect()

    interpretation = detector.build_interpretation(result.current_regime)
    conf_label = detector.confidence_label(result.confidence)

    print("\n" + "=" * 72)
    print("               WORLD MARKET REGIME DETECTOR")
    print("=" * 72)

    print("\n[ CURRENT MARKET SNAPSHOT ]")
    print(f"Date           : {result.latest_date}")
    print(f"Current Regime : {result.current_regime}")
    print(f"Hidden State   : State {result.state_id}")
    print(f"Confidence     : {conf_label} ({result.confidence:.2%})")
    print(f"Interpretation : {interpretation}")

    print("\n" + "-" * 72)
    print("[ REGIME SUMMARY ]")
    print("-" * 72)

    summary_display = result.summary_table.copy()
    summary_display["avg_return"] = (summary_display["avg_return"] * 100).round(2).astype(str) + "%"
    summary_display["avg_vol"] = (summary_display["avg_vol"] * 100).round(2).astype(str) + "%"
    summary_display["avg_trend"] = (summary_display["avg_trend"] * 100).round(2).astype(str) + "%"
    summary_display["avg_vix"] = summary_display["avg_vix"].round(2)
    print(summary_display.to_string(index=False))

    print("\n" + "-" * 72)
    print("[ STATE TRANSITION MATRIX ]")
    print("-" * 72)
    print(result.transition_matrix.round(3).to_string())

    print("\n" + "-" * 72)
    print("[ OUTPUT FILES ]")
    print("-" * 72)
    print("results/latest_regime.json")
    print("results/regime_chart.png")

    print("\n" + "=" * 72)

if __name__ == "__main__":
    main()
    
    
    