# 🔗 Pairs Trading with Cointegration

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Statistics](https://img.shields.io/badge/Method-Cointegration-green)
![Trading](https://img.shields.io/badge/Strategy-Pairs%20Trading-orange)
![Finance](https://img.shields.io/badge/Domain-Quantitative%20Finance-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

A quantitative finance project that builds and **backtests a mean-reversion pairs trading strategy** on two related assets, from raw prices all the way through to performance metrics.

The pipeline estimates a hedge ratio, tests the pair for cointegration, constructs the spread and its z-score, generates long/short signals, and backtests the resulting strategy with standard risk metrics (Sharpe, drawdown, win rate).

---

# 🧠 Motivation

Pairs trading is a classic **statistical arbitrage** strategy used in quantitative finance.

The idea is simple:

- choose two related assets
- test whether they are statistically linked
- measure deviations between them
- trade when the spread moves abnormally far from its historical relationship

---

# 📊 Assets Used

The default run uses:

- **KO** — Coca-Cola
- **PEP** — PepsiCo

Both companies operate in a similar industry and often exhibit related price behavior, which makes them a common textbook pair. The tickers are configurable via `PairsTradingPipeline(ticker_y=..., ticker_x=...)`.

---

# ⚙️ How the Pipeline Works

`src/pairs_trading_pipeline.py` runs end to end:

1. Download adjusted close prices for both assets
2. Estimate the **hedge ratio** via OLS (`LinearRegression`)
3. Run the **Engle-Granger cointegration test** (`statsmodels.tsa.stattools.coint`)
4. Build the spread `y − β·x` and its rolling z-score (`lookback = 20`)
5. Generate positions from z-score thresholds (entry ±2.0, exit ±0.5, stop ±3.5)
6. Backtest the spread strategy and compute performance metrics
7. Save datasets, a results summary, a backtest CSV, and charts

---

# 📈 Backtest Results

Results from the default 10-year KO/PEP run (saved to `results/summary.json`):

| Metric | Value |
|--------|-------|
| Cointegration p-value | 0.954 |
| Hedge ratio (β) | 0.353 |
| Total return | 17.9% |
| Annualized return | 1.7% |
| Annualized volatility | 9.4% |
| Sharpe ratio | 0.18 |
| Max drawdown | −26.7% |
| Win rate | 52.1% |
| Trades | 198 |

> ⚠️ **Interpretation:** the Engle-Granger p-value of ~0.95 means we **cannot reject
> the null of no cointegration** — over this window KO and PEP are not a statistically
> cointegrated pair. The backtest is therefore best read as a **demonstration of the
> methodology** (hedge ratio → spread → z-score signals → risk metrics) rather than
> evidence of a tradeable edge. Re-running on a genuinely cointegrated pair is the
> natural next step.

---

# 📁 Project Structure

```
pairs_trading_cointegration/
│
├── src/
│   ├── __init__.py
│   └── pairs_trading_pipeline.py
│
├── data/
│   ├── generate_pairs_dataset.py       # standalone dataset builder
│   ├── raw/
│   │   └── ko_pep_prices.csv
│   └── processed/
│       └── ko_pep_spread_dataset.csv
│
├── notebooks/
│   └── pairs_trading_exploration.ipynb
│
├── results/
│   ├── summary.json
│   ├── pairs_trading_backtest.csv
│   ├── spread_zscore.png
│   └── equity_curve.png
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

From the repository root:

```bash
git clone https://github.com/abhipabhi/quant-ml-lab.git
cd quant-ml-lab
pip install -r requirements.txt
```

---

# ▶️ Running the Pipeline

Run the full strategy pipeline from the repository root:

```bash
python research/pairs_trading_cointegration/src/pairs_trading_pipeline.py
```

To only regenerate the exploratory spread dataset (no backtest), run the standalone
generator from inside the project directory:

```bash
cd research/pairs_trading_cointegration
python data/generate_pairs_dataset.py
```

---

# 📊 Generated Outputs

| Output | Description |
|--------|-------------|
| `results/summary.json` | Cointegration p-value, hedge ratio, and all performance metrics |
| `results/pairs_trading_backtest.csv` | Full backtest: spread, z-score, positions, returns, drawdown |
| `results/spread_zscore.png` | Spread z-score with entry/exit thresholds |
| `results/equity_curve.png` | Strategy cumulative return |
| `data/raw/ko_pep_prices.csv` | Aligned daily close prices |
| `data/processed/ko_pep_spread_dataset.csv` | Exploratory spread dataset |

---

# 🔬 Potential Extensions

Possible next steps include:

- rolling / time-varying hedge ratio estimation
- selecting pairs by screening a universe for cointegration
- transaction costs and slippage modeling
- half-life of mean reversion for adaptive lookbacks
- Kalman-filter spread estimation

---

# ⚠️ Disclaimer

This project is intended **for research and educational purposes only**.

It should **not be considered financial advice**.

---

# 👨‍💻 Author

**Abhi Patidar**
