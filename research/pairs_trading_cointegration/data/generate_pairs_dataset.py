import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

TICKERS = {
    "KO": "KO",
    "PEP": "PEP",
}

PERIOD = "10y"
INTERVAL = "1d"


def fetch_pair_data() -> pd.DataFrame:
    """Download historical price data for the pair."""

    close_frames = []

    for name, ticker in TICKERS.items():
        print(f"Downloading {name} ({ticker})...")

        df = yf.download(
            ticker,
            period=PERIOD,
            interval=INTERVAL,
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

    return prices


def build_spread_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    """Create processed dataset for pairs trading research."""

    df = prices.copy()

    df["ko_return"] = np.log(df["KO_close"] / df["KO_close"].shift(1))
    df["pep_return"] = np.log(df["PEP_close"] / df["PEP_close"].shift(1))

    # Simple spread for version 1
    df["spread"] = df["KO_close"] - df["PEP_close"]

    df["spread_mean_20"] = df["spread"].rolling(20).mean()
    df["spread_std_20"] = df["spread"].rolling(20).std()
    df["spread_zscore"] = (
        (df["spread"] - df["spread_mean_20"]) / df["spread_std_20"]
    )

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df.index.name = "Date"

    return df


def save_datasets(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> None:
    """Save raw and processed datasets."""

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "ko_pep_prices.csv"
    processed_path = processed_dir / "ko_pep_spread_dataset.csv"

    raw_df.to_csv(raw_path)
    processed_df.to_csv(processed_path)

    print("\nSaved datasets:")
    print(raw_path)
    print(processed_path)


def main() -> None:
    print("\nGenerating pairs trading dataset...\n")

    prices = fetch_pair_data()
    dataset = build_spread_dataset(prices)
    save_datasets(prices, dataset)

    print("\nDataset summary:")
    print(f"Raw rows       : {len(prices)}")
    print(f"Processed rows : {len(dataset)}")
    print(f"Processed cols : {list(dataset.columns)}")


if __name__ == "__main__":
    main()