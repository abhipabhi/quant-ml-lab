import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path


TICKERS = {
    "US": "^GSPC",       # S&P 500
    "EU": "^STOXX50E",   # Euro Stoxx 50
    "INDIA": "^NSEI",    # Nifty 50
    "JAPAN": "^N225",    # Nikkei 225
    "VIX": "^VIX"
}

PERIOD = "10y"
INTERVAL = "1d"


def fetch_market_data():
    """Download historical market prices"""

    close_frames = []

    for market, ticker in TICKERS.items():
        print(f"Downloading {market} ({ticker})")

        df = yf.download(
            ticker,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            raise ValueError(f"Failed to download {ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            series = df.xs("Close", axis=1, level=0).iloc[:, 0]
        else:
            series = df["Close"]

        series = series.rename(market)
        close_frames.append(series)

    prices = pd.concat(close_frames, axis=1).dropna()

    return prices


def build_features(prices):
    """Generate regime detection features"""

    log_returns = np.log(prices / prices.shift(1)).dropna()

    feat = pd.DataFrame(index=log_returns.index)

    equity_cols = [c for c in log_returns.columns if c != "VIX"]

    feat["global_equity_return"] = log_returns[equity_cols].mean(axis=1)

    feat["cross_sectional_dispersion"] = log_returns[equity_cols].std(axis=1)

    feat["realized_vol_20"] = feat["global_equity_return"].rolling(20).std()

    avg_price = prices[equity_cols].mean(axis=1)

    ma20 = avg_price.rolling(20).mean()
    ma100 = avg_price.rolling(100).mean()

    feat["trend_ratio"] = ma20 / ma100 - 1.0

    feat["vix_level"] = prices["VIX"]

    feat["vix_return_5"] = np.log(prices["VIX"] / prices["VIX"].shift(5))

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

    return feat


def save_datasets(prices, features):
    """Save raw and processed datasets"""

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_file = raw_dir / "world_market_prices.csv"
    processed_file = processed_dir / "regime_features.csv"

    prices.to_csv(raw_file)
    features.to_csv(processed_file)

    print("\nSaved files:")
    print(raw_file)
    print(processed_file)


def main():

    print("\nFetching market data...")
    prices = fetch_market_data()

    print("Generating features...")
    features = build_features(prices)

    print("Saving datasets...")
    save_datasets(prices, features)

    print("\nDone.")
    print(f"Raw data rows: {len(prices)}")
    print(f"Feature rows: {len(features)}")


if __name__ == "__main__":
    main()