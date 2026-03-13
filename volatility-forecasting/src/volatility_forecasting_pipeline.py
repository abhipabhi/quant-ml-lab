from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


@dataclass
class ForecastingResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame


class VolatilityForecastingPipeline:
    def __init__(
        self,
        ticker: str = "^GSPC",
        period: str = "10y",
        interval: str = "1d",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.test_size = test_size
        self.random_state = random_state

        self.raw_data: pd.DataFrame | None = None
        self.dataset: pd.DataFrame | None = None
        self.feature_columns: list[str] | None = None

    def fetch_market_data(self) -> pd.DataFrame:
        df = yf.download(
            self.ticker,
            period=self.period,
            interval=self.interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df.empty:
            raise ValueError(f"Failed to download data for {self.ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            close_data = df.xs("Close", axis=1, level=0)
            if close_data.empty:
                raise ValueError("No Close data found in downloaded dataset")
            close_series = close_data.iloc[:, 0]
        else:
            if "Close" not in df.columns:
                raise ValueError("No Close column found in downloaded dataset")
            close_series = df["Close"]

        prices = close_series.rename("close").to_frame()
        prices.index.name = "Date"

        self.raw_data = prices
        return prices

    def build_dataset(self, prices: pd.DataFrame) -> pd.DataFrame:
        df = prices.copy()

        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["abs_return"] = df["log_return"].abs()

        df["vol_5"] = df["log_return"].rolling(5).std()
        df["vol_10"] = df["log_return"].rolling(10).std()
        df["vol_20"] = df["log_return"].rolling(20).std()

        df["mean_return_5"] = df["log_return"].rolling(5).mean()
        df["mean_return_10"] = df["log_return"].rolling(10).mean()

        df["momentum_5"] = df["close"].pct_change(5)
        df["momentum_10"] = df["close"].pct_change(10)

        # Future 5-day realized volatility target
        df["target_vol_5"] = df["log_return"].rolling(5).std().shift(-5)

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df.index.name = "Date"

        self.feature_columns = [
            "log_return",
            "abs_return",
            "vol_5",
            "vol_10",
            "vol_20",
            "mean_return_5",
            "mean_return_10",
            "momentum_5",
            "momentum_10",
        ]

        self.dataset = df
        return df

    def save_datasets(self, prices: pd.DataFrame, dataset: pd.DataFrame) -> None:
        raw_dir = Path("data/raw")
        processed_dir = Path("data/processed")
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        prices.to_csv(raw_dir / "sp500_prices.csv")
        dataset.to_csv(processed_dir / "volatility_features.csv")

    def time_split(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_idx = int(len(dataset) * (1 - self.test_size))
        train = dataset.iloc[:split_idx].copy()
        test = dataset.iloc[split_idx:].copy()
        return train, test

    def baseline_forecast(self, test_df: pd.DataFrame) -> pd.Series:
        # Simple baseline: predict future vol by current 20-day realized vol
        return test_df["vol_20"].copy()

    def garch_forecast(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
        train_returns = train_df["log_return"] * 100.0
        full_returns = pd.concat([train_df["log_return"], test_df["log_return"]]) * 100.0

        predictions = []
        start = len(train_df)
        end = len(train_df) + len(test_df)

        for i in range(start, end):
            rolling_train = full_returns.iloc[:i].dropna()

            model = arch_model(
                rolling_train,
                mean="Zero",
                vol="GARCH",
                p=1,
                q=1,
                dist="normal",
            )
            fitted = model.fit(disp="off")

            forecast = fitted.forecast(horizon=5, reindex=False)
            daily_variances = forecast.variance.values[-1]

            # Convert avg daily variance over 5 days into volatility
            pred_vol = np.sqrt(np.mean(daily_variances)) / 100.0
            predictions.append(pred_vol)

        return pd.Series(predictions, index=test_df.index, name="garch_pred")

    def random_forest_forecast(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
        if self.feature_columns is None:
            raise ValueError("Feature columns not initialized")

        X_train = train_df[self.feature_columns]
        y_train = train_df["target_vol_5"]

        X_test = test_df[self.feature_columns]

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        return pd.Series(preds, index=test_df.index, name="rf_pred")

    def evaluate_model(self, actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        mae = float(mean_absolute_error(actual, predicted))
        return {"RMSE": rmse, "MAE": mae}

    def save_results(self, result: ForecastingResult) -> None:
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        result.metrics.to_csv(results_dir / "model_metrics.csv", index=False)
        result.predictions.to_csv(results_dir / "volatility_predictions.csv")

        payload = {
            "best_model_by_rmse": result.metrics.sort_values("RMSE").iloc[0]["Model"],
            "best_model_by_mae": result.metrics.sort_values("MAE").iloc[0]["Model"],
        }

        with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

    def plot_results(self, predictions_df: pd.DataFrame) -> None:
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(14, 7))
        plt.plot(predictions_df.index, predictions_df["actual"], label="Actual Future Volatility", linewidth=2)
        plt.plot(predictions_df.index, predictions_df["baseline"], label="Baseline (Vol 20)", linewidth=1.5)
        plt.plot(predictions_df.index, predictions_df["garch"], label="GARCH(1,1)", linewidth=1.5)
        plt.plot(predictions_df.index, predictions_df["random_forest"], label="Random Forest", linewidth=1.5)
        plt.title("Volatility Forecasting: Actual vs Predicted")
        plt.xlabel("Date")
        plt.ylabel("5-Day Future Realized Volatility")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "forecast_comparison.png", dpi=300)
        plt.close()

    def run(self) -> ForecastingResult:
        prices = self.fetch_market_data()
        dataset = self.build_dataset(prices)
        self.save_datasets(prices, dataset)

        train_df, test_df = self.time_split(dataset)

        actual = test_df["target_vol_5"].copy()
        baseline_pred = self.baseline_forecast(test_df)
        garch_pred = self.garch_forecast(train_df, test_df)
        rf_pred = self.random_forest_forecast(train_df, test_df)

        predictions_df = pd.DataFrame(
            {
                "actual": actual,
                "baseline": baseline_pred,
                "garch": garch_pred,
                "random_forest": rf_pred,
            },
            index=test_df.index,
        )

        metrics = []
        metrics.append({"Model": "Baseline (Vol 20)", **self.evaluate_model(actual, baseline_pred)})
        metrics.append({"Model": "GARCH(1,1)", **self.evaluate_model(actual, garch_pred)})
        metrics.append({"Model": "Random Forest", **self.evaluate_model(actual, rf_pred)})

        metrics_df = pd.DataFrame(metrics).sort_values("RMSE").reset_index(drop=True)

        result = ForecastingResult(metrics=metrics_df, predictions=predictions_df)

        self.save_results(result)
        self.plot_results(predictions_df)

        return result


def main() -> None:
    pipeline = VolatilityForecastingPipeline()
    result = pipeline.run()

    print("\n" + "=" * 72)
    print("                  VOLATILITY FORECASTING PIPELINE")
    print("=" * 72)

    print("\n[ MODEL PERFORMANCE ]")
    print(result.metrics.round(6).to_string(index=False))

    print("\n[ OUTPUT FILES ]")
    print("data/raw/sp500_prices.csv")
    print("data/processed/volatility_features.csv")
    print("results/model_metrics.csv")
    print("results/volatility_predictions.csv")
    print("results/forecast_comparison.png")
    print("results/summary.json")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()