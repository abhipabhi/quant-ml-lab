# 🌍 Global Market Regime Detection

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-HMM-green)
![Finance](https://img.shields.io/badge/Domain-Quantitative%20Finance-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning system that detects the **current global financial market regime** using a Hidden Markov Model trained on multi-market financial time series.

The model analyzes historical behavior across major equity indices and volatility indicators to classify the current market environment into interpretable regimes such as:

- **Bull / Low-Vol**
- **Bull / High-Vol**
- **Neutral / Stable**
- **Bear / Risk-Off**

The pipeline automatically:

1. Fetches global market data
2. Engineers financial features
3. Trains an unsupervised Hidden Markov Model
4. Detects the current market regime
5. Produces structured outputs and visualizations

---

# 📊 Example Output

```
========================================================================
               WORLD MARKET REGIME DETECTOR
========================================================================

[ CURRENT MARKET SNAPSHOT ]
Date           : 2026-03-12
Current Regime : Bull / High-Vol
Confidence     : Very High (100%)

Interpretation : Markets are positive but volatile, indicating
risk-on conditions with elevated uncertainty.
```

The pipeline also generates:

```
results/latest_regime.json
results/regime_chart.png
```

---

# 🧠 Methodology

This project follows a typical **quantitative research pipeline**.

## 1. Data Collection

Market data is fetched dynamically using **Yahoo Finance** via `yfinance`.

The model uses a basket of global financial indicators:

| Asset | Description |
|------|------|
| S&P 500 (`^GSPC`) | US equity market |
| Euro Stoxx 50 (`^STOXX50E`) | European equities |
| Nifty 50 (`^NSEI`) | Indian equities |
| Nikkei 225 (`^N225`) | Japanese equities |
| VIX (`^VIX`) | Market volatility / risk sentiment |

These assets provide a **proxy representation of global risk conditions**.

---

## 2. Feature Engineering

Rather than using raw prices, the model uses derived financial features.

| Feature | Description |
|------|------|
| Global Equity Return | Average return across equity indices |
| Cross-Market Dispersion | Standard deviation of market returns |
| 20-Day Realized Volatility | Rolling volatility of global returns |
| Trend Ratio | 20-day vs 100-day moving average ratio |
| VIX Level | Market implied volatility |
| VIX 5-Day Return | Short-term volatility momentum |

These features capture **trend, volatility, and systemic risk conditions**.

---

## 3. Hidden Markov Model

A **Gaussian Hidden Markov Model (HMM)** is trained on the feature set.

The model assumes:

- Financial markets switch between **hidden regimes**
- Each regime produces distinct statistical behavior
- Regimes transition probabilistically over time

The HMM automatically learns:

- regime-specific feature distributions
- transition probabilities between regimes
- the most likely regime for each observation

---

## 4. Regime Classification

The detected hidden states are mapped to interpretable regimes.

| Regime | Description |
|------|------|
| Bull / Low-Vol | Upward trend with low volatility |
| Bull / High-Vol | Positive market conditions with elevated volatility |
| Neutral / Stable | Sideways or mildly positive market |
| Bear / Risk-Off | Declining market with elevated volatility |

---

# 📁 Project Structure

```
market-regime-detection/
│
├── src/
│   ├── __init__.py
│   └── current_world_market_regime.py
│
├── data/
│   ├── raw/
│   │   └── world_market_prices.csv
│   │
│   └── processed/
│       └── regime_features.csv
│
├── notebooks/
│   └── regime_exploration.ipynb
│
├── results/
│   ├── latest_regime.json
│   └── regime_chart.png
│
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<username>/market-regime-detection.git
cd market-regime-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Regime Detector

Run the main pipeline:

```bash
python src/current_world_market_regime.py
```

The script will:

1. Download historical market data
2. Generate financial features
3. Train the Hidden Markov Model
4. Detect the current regime
5. Save results to disk

---

# 📈 Generated Outputs

## JSON Summary

```
results/latest_regime.json
```

Example:

```json
{
    "date": "2026-03-12",
    "current_regime": "Bull / High-Vol",
    "confidence": 1.0,
    "confidence_label": "Very High"
}
```

---

## Regime Visualization

```
results/regime_chart.png
```

This chart visualizes the detected regimes across time and overlays them on the global equity index.

---

# 🔬 Research Extensions

Potential improvements include:

- macroeconomic indicators (rates, inflation)
- commodity markets
- currency markets
- regime forecasting
- Bayesian regime models
- reinforcement learning portfolio allocation

---

# ⚠️ Disclaimer

This project is intended **for research and educational purposes only**.

It should **not be used as financial advice or as the sole basis for investment decisions**.

---

# 👨‍💻 Author

Abhi Patidar
