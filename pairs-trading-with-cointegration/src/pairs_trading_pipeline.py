from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")


@dataclass
class PairsTradingResult:
    pair_name: str
    cointegration_pvalue: float
    hedge_ratio: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades: int
    predictions: pd.DataFrame


class PairsTradingPipeline:
    def __init__(
        self,
        ticker_y: str = "KO",
        ticker_x: str = "PEP",
        period: str = "10y",
        interval: str = "1d",
        lookback: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.5,
    ) -> None:
        self.ticker_y = ticker_y
        self.ticker_x = ticker_x
        self.period = period
        self.interval = interval
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z

        self.raw_data: pd.DataFrame | None = None
        self.dataset: pd.DataFrame | None = None

    def fetch_pair_data(self) -> pd.DataFrame:
        close_frames = []

        for name, ticker in {self.ticker_y: self.ticker_y, self.ticker_x: self.ticker_x}.items():
            df = yf.download(
                ticker,
                period=self.period,
                interval=self.interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if df.empty:
                raise ValueError(f"Failed to download data for {ticker}")

            if isinstance(df.columns, pd.MultiIndex):
                close_data = df.xs("Close", axis=1, level=0)
                if close_data.empty:
                    raise ValueError(f"No Close data found for {ticker}")
                close_series = close_data.iloc[:, 0]
            else:
                if "Close" not in df.columns:
                    raise ValueError(f"No Close column found for {ticker}")
                close_series = df["Close"]

            close_series = close_series.rename(f"{name}_close")
            close_frames.append(close_series)

        prices = pd.concat(close_frames, axis=1).dropna()
        prices.index.name = "Date"

        self.raw_data = prices
        return prices

    def estimate_hedge_ratio(self, prices: pd.DataFrame) -> float:
        y = prices[f"{self.ticker_y}_close"].values.reshape(-1, 1)
        x = prices[f"{self.ticker_x}_close"].values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(x, y)

        hedge_ratio = float(model.coef_[0][0])
        return hedge_ratio

    def build_spread_dataset(self, prices: pd.DataFrame, hedge_ratio: float) -> pd.DataFrame:
        df = prices.copy()

        y_col = f"{self.ticker_y}_close"
        x_col = f"{self.ticker_x}_close"

        df[f"{self.ticker_y.lower()}_return"] = np.log(df[y_col] / df[y_col].shift(1))
        df[f"{self.ticker_x.lower()}_return"] = np.log(df[x_col] / df[x_col].shift(1))

        df["spread"] = df[y_col] - hedge_ratio * df[x_col]
        df["spread_mean"] = df["spread"].rolling(self.lookback).mean()
        df["spread_std"] = df["spread"].rolling(self.lookback).std()
        df["spread_zscore"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df.index.name = "Date"

        self.dataset = df
        return df

    def test_cointegration(self, prices: pd.DataFrame) -> float:
        y = prices[f"{self.ticker_y}_close"]
        x = prices[f"{self.ticker_x}_close"]
        _, pvalue, _ = coint(y, x)
        return float(pvalue)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # position: +1 = long spread, -1 = short spread, 0 = flat
        position = np.zeros(len(data))

        z = data["spread_zscore"].values

        for i in range(1, len(data)):
            prev_pos = position[i - 1]
            current_z = z[i]

            if prev_pos == 0:
                if current_z > self.entry_z:
                    position[i] = -1
                elif current_z < -self.entry_z:
                    position[i] = 1
                else:
                    position[i] = 0

            elif prev_pos == 1:
                if current_z >= -self.exit_z or current_z < -self.stop_z:
                    position[i] = 0
                else:
                    position[i] = 1

            elif prev_pos == -1:
                if current_z <= self.exit_z or current_z > self.stop_z:
                    position[i] = 0
                else:
                    position[i] = -1

        data["position"] = position
        data["position_change"] = data["position"].diff().fillna(0)

        return data

    def backtest_strategy(self, df: pd.DataFrame, hedge_ratio: float) -> pd.DataFrame:
        data = df.copy()

        y_ret = data[f"{self.ticker_y.lower()}_return"]
        x_ret = data[f"{self.ticker_x.lower()}_return"]

        # Spread return approximation
        data["strategy_return"] = (
            data["position"].shift(1).fillna(0) * (y_ret - hedge_ratio * x_ret)
        )

        data["cumulative_return"] = (1 + data["strategy_return"]).cumprod() - 1
        data["rolling_peak"] = (1 + data["cumulative_return"]).cummax()
        equity_curve = 1 + data["cumulative_return"]
        data["drawdown"] = equity_curve / equity_curve.cummax() - 1

        return data

    def compute_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        returns = df["strategy_return"].dropna()

        if len(returns) == 0:
            raise ValueError("No strategy returns generated.")

        total_return = float((1 + returns).prod() - 1)
        annualized_return = float((1 + total_return) ** (252 / len(returns)) - 1) if len(returns) > 0 else 0.0
        annualized_volatility = float(returns.std() * np.sqrt(252))

        if annualized_volatility > 0:
            sharpe_ratio = float(annualized_return / annualized_volatility)
        else:
            sharpe_ratio = 0.0

        max_drawdown = float(df["drawdown"].min())

        trade_returns = returns[returns != 0]
        win_rate = float((trade_returns > 0).mean()) if len(trade_returns) > 0 else 0.0

        trades = int((df["position_change"].abs() > 0).sum())

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "trades": trades,
        }

    def save_datasets(self, prices: pd.DataFrame, dataset: pd.DataFrame) -> None:
        raw_dir = Path("data/raw")
        processed_dir = Path("data/processed")
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        prices.to_csv(raw_dir / f"{self.ticker_y.lower()}_{self.ticker_x.lower()}_prices.csv")
        dataset.to_csv(processed_dir / f"{self.ticker_y.lower()}_{self.ticker_x.lower()}_spread_dataset.csv")

    def save_results(self, result: PairsTradingResult) -> None:
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        result.predictions.to_csv(results_dir / "pairs_trading_backtest.csv")

        summary = {
            "pair_name": result.pair_name,
            "cointegration_pvalue": result.cointegration_pvalue,
            "hedge_ratio": result.hedge_ratio,
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "annualized_volatility": result.annualized_volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "trades": result.trades,
        }

        with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

    def plot_results(self, df: pd.DataFrame) -> None:
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(14, 6))
        plt.plot(df.index, df["spread_zscore"], label="Spread Z-Score", linewidth=1.5)
        plt.axhline(self.entry_z, color="red", linestyle="--", label="Entry Threshold")
        plt.axhline(-self.entry_z, color="green", linestyle="--")
        plt.axhline(self.exit_z, color="orange", linestyle=":", label="Exit Threshold")
        plt.axhline(-self.exit_z, color="orange", linestyle=":")
        plt.axhline(0, color="black", linewidth=1)
        plt.title(f"{self.ticker_y}/{self.ticker_x} Spread Z-Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "spread_zscore.png", dpi=300)
        plt.close()

        plt.figure(figsize=(14, 6))
        plt.plot(df.index, df["cumulative_return"], label="Strategy Cumulative Return", linewidth=2)
        plt.axhline(0, color="black", linewidth=1)
        plt.title(f"{self.ticker_y}/{self.ticker_x} Pairs Trading Equity Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "equity_curve.png", dpi=300)
        plt.close()

    def run(self) -> PairsTradingResult:
        prices = self.fetch_pair_data()
        hedge_ratio = self.estimate_hedge_ratio(prices)
        pvalue = self.test_cointegration(prices)

        dataset = self.build_spread_dataset(prices, hedge_ratio)
        self.save_datasets(prices, dataset)

        signals_df = self.generate_signals(dataset)
        backtest_df = self.backtest_strategy(signals_df, hedge_ratio)

        metrics = self.compute_metrics(backtest_df)

        result = PairsTradingResult(
            pair_name=f"{self.ticker_y}/{self.ticker_x}",
            cointegration_pvalue=pvalue,
            hedge_ratio=hedge_ratio,
            total_return=metrics["total_return"],
            annualized_return=metrics["annualized_return"],
            annualized_volatility=metrics["annualized_volatility"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
            trades=metrics["trades"],
            predictions=backtest_df,
        )

        self.save_results(result)
        self.plot_results(backtest_df)

        return result


def main() -> None:
    pipeline = PairsTradingPipeline()
    result = pipeline.run()

    print("\n" + "=" * 72)
    print("                    PAIRS TRADING PIPELINE")
    print("=" * 72)

    print(f"\nPair                  : {result.pair_name}")
    print(f"Cointegration p-value : {result.cointegration_pvalue:.6f}")
    print(f"Hedge Ratio           : {result.hedge_ratio:.4f}")

    print("\n[ BACKTEST PERFORMANCE ]")
    print(f"Total Return          : {result.total_return:.2%}")
    print(f"Annualized Return     : {result.annualized_return:.2%}")
    print(f"Annualized Volatility : {result.annualized_volatility:.2%}")
    print(f"Sharpe Ratio          : {result.sharpe_ratio:.4f}")
    print(f"Max Drawdown          : {result.max_drawdown:.2%}")
    print(f"Win Rate              : {result.win_rate:.2%}")
    print(f"Trades                : {result.trades}")

    print("\n[ OUTPUT FILES ]")
    print(f"data/raw/{pipeline.ticker_y.lower()}_{pipeline.ticker_x.lower()}_prices.csv")
    print(f"data/processed/{pipeline.ticker_y.lower()}_{pipeline.ticker_x.lower()}_spread_dataset.csv")
    print("results/pairs_trading_backtest.csv")
    print("results/summary.json")
    print("results/spread_zscore.png")
    print("results/equity_curve.png")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()