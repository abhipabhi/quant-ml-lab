import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

TICKER = "^GSPC"   # S&P 500
PERIOD = "10y"
INTERVAL = "1d"


def fetch_market_data() -> pd.DataFrame:
    """Download historical price data and return close prices."""

    print("Downloading market data...")

    df = yf.download(
        TICKER,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if df.empty:
        raise ValueError("Failed to download market data")

    # Handle both normal and MultiIndex yfinance outputs
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

    return prices


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Generate features and target for volatility forecasting."""

    df = prices.copy()

    # Log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Absolute return
    df["abs_return"] = df["log_return"].abs()

    # Rolling volatility features
    df["vol_5"] = df["log_return"].rolling(5).std()
    df["vol_10"] = df["log_return"].rolling(10).std()
    df["vol_20"] = df["log_return"].rolling(20).std()

    # Rolling average returns
    df["mean_return_5"] = df["log_return"].rolling(5).mean()
    df["mean_return_10"] = df["log_return"].rolling(10).mean()

    # Momentum features
    df["momentum_5"] = df["close"].pct_change(5)
    df["momentum_10"] = df["close"].pct_change(10)

    # Future 5-day realized volatility target
    # Uses next 5 trading days as target
    df["target_vol_5"] = df["log_return"].rolling(5).std().shift(-5)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df.index.name = "Date"

    return df


def save_datasets(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> None:
    """Save raw and processed datasets to CSV."""

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "sp500_prices.csv"
    processed_path = processed_dir / "volatility_features.csv"

    raw_df.to_csv(raw_path)
    processed_df.to_csv(processed_path)

    print("\nSaved datasets:")
    print(raw_path)
    print(processed_path)


def main() -> None:
    prices = fetch_market_data()

    print("Generating features...")
    features = build_features(prices)

    print("Saving datasets...")
    save_datasets(prices, features)

    print("\nDataset summary:")
    print(f"Raw rows       : {len(prices)}")
    print(f"Processed rows : {len(features)}")
    print(f"Processed cols : {list(features.columns)}")


if __name__ == "__main__":
    main()